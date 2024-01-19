import os, json, argparse, yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from diffusers import DDIMScheduler

from dataset import VCDecLPCDataset, VCDecLPCBatchCollate, VCDecLPCTest
from models.unet import UNetVC
from modules.BigVGAN.inference import load_model
from utils import save_plot, save_audio
from utils import minmax_norm_diff, reverse_minmax_norm_diff


parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, default='config/DiffLPC.yaml')

parser.add_argument('-seed', type=int, default=98)
parser.add_argument('-amp', type=bool, default=True)
parser.add_argument('-compile', type=bool, default=False)

parser.add_argument('-data_dir', type=str, default='../data/')
parser.add_argument('-lpc_dir', type=str, default='lpc')
parser.add_argument('-vocoder_dir', type=str, default='modules/BigVGAN/ckpt/bigvgan_base_22khz_80band/g_05000000')

parser.add_argument('-train_frames', type=int, default=128)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-test_size', type=int, default=1)
parser.add_argument('-num_workers', type=int, default=4)
parser.add_argument('-lr', type=float, default=5e-5)
parser.add_argument('-weight_decay', type=int, default=1e-6)

parser.add_argument('-epochs', type=int, default=32)
parser.add_argument('-save_every', type=int, default=2)
parser.add_argument('-log_step', type=int, default=200)
parser.add_argument('-log_dir', type=str, default='logs_dec_lpc')
parser.add_argument('-ckpt_dir', type=str, default='ckpt')

args = parser.parse_args()
args.save_ori = True
config = yaml.load(open(args.config), Loader=yaml.FullLoader)
mel_cfg = config['logmel']
ddpm_cfg = config['ddpm']
unet_cfg = config['unet']

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
    else:
       args.device = 'cpu'

    if os.path.exists(args.log_dir) is False:
        os.makedirs(args.log_dir)

    if os.path.exists(args.ckpt_dir) is False:
        os.makedirs(args.ckpt_dir)

    print('Initializing vocoder...')
    hifigan, cfg = load_model(args.vocoder_dir, device=args.device)

    print('Initializing data loaders...')
    train_set = VCDecLPCDataset(args.data_dir, subset='train', content_dir=args.lpc_dir)
    collate_fn = VCDecLPCBatchCollate(args.train_frames)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True)

    val_set = VCDecLPCTest(args.data_dir, content_dir=args.lpc_dir)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    print('Initializing and loading models...')
    model = UNetVC(**unet_cfg).to(args.device)
    print('Number of parameters = %.2fm\n' % (model.nparams / 1e6))

    # prepare DPM scheduler
    noise_scheduler = DDIMScheduler(num_train_timesteps=ddpm_cfg['num_train_steps'])

    print('Initializing optimizers...')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    if args.compile:
        model = torch.compile(model)

    print('Start training.')
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch: {epoch} [iteration: {global_step}]')
        model.train()
        losses = []

        for step, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            # make spectrogram range from -1 to 1
            mel = batch['mel1'].to(args.device)
            mel = minmax_norm_diff(mel, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

            if unet_cfg["use_ref_t"]:
                mel_ref = batch['mel2'].to(args.device)
                mel_ref = minmax_norm_diff(mel_ref, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            else:
                mel_ref = None

            f0 = batch['f0_1'].to(args.device)

            mean = batch['content1'].to(args.device)
            mean = minmax_norm_diff(mean, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

            noise = torch.randn(mel.shape).to(args.device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps,
                                      (args.batch_size,),
                                      device=args.device, ).long()

            noisy_mel = noise_scheduler.add_noise(mel, noise, timesteps)

            if args.amp:
                with autocast():
                    noise_pred = model(x=noisy_mel, mean=mean, f0=f0, t=timesteps, ref=mel_ref, embed=None)
                    loss = F.mse_loss(noise_pred, noise)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                noise_pred = model(x=noisy_mel, mean=mean, f0=f0, t=timesteps, ref=mel_ref, embed=None)
                loss = F.mse_loss(noise_pred, noise)
                # Backward propagation
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            global_step += 1

            if global_step % args.log_step == 0:
                losses = np.asarray(losses)
                # msg = 'Epoch %d: loss = %.4f\n' % (epoch, np.mean(losses))
                msg = '\nEpoch: [{}][{}]\t' \
                      'Batch: [{}][{}]\tLoss: {:.6f}\n'.format(epoch,
                                                               args.epochs,
                                                               step+1,
                                                               len(train_loader),
                                                               np.mean(losses))
                with open(f'{args.log_dir}/train_dec.log', 'a') as f:
                    f.write(msg)
                losses = []

        if epoch % args.save_every > 0:
            continue

        print('Saving model...\n')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{args.ckpt_dir}/lpc_vc_{epoch}.pt")

        print('Inference...\n')
        noise = None
        noise_scheduler.set_timesteps(ddpm_cfg['inference_steps'])
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # optimizer.zero_grad()
                generator = torch.Generator(device=args.device).manual_seed(args.seed)

                mel = batch['mel1'].to(args.device)
                mel = minmax_norm_diff(mel, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

                if unet_cfg["use_ref_t"]:
                    mel_ref = batch['mel2'].to(args.device)
                    mel_ref = minmax_norm_diff(mel_ref, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
                else:
                    mel_ref = None

                f0 = batch['f0_1'].to(args.device)
                embed = batch['embed'].to(args.device)

                mean = batch['content1'].to(args.device)
                mean = minmax_norm_diff(mean, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

                # make spectrogram range from -1 to 1
                if noise is None:
                    noise = torch.randn(mel.shape,
                                        generator=generator,
                                        device=args.device,
                                        )
                pred = noise

                for t in noise_scheduler.timesteps:
                    pred = noise_scheduler.scale_model_input(pred, t)
                    model_output = model(x=pred, mean=mean, f0=f0, t=t, ref=mel_ref, embed=None)
                    pred = noise_scheduler.step(model_output=model_output,
                                                timestep=t,
                                                sample=pred,
                                                eta=ddpm_cfg['eta'], generator=generator).prev_sample


                if os.path.exists(f'{args.log_dir}/audio/{i}/') is False:
                    os.makedirs(f'{args.log_dir}/audio/{i}/')
                    os.makedirs(f'{args.log_dir}/pic/{i}/')

                # save pred
                pred = reverse_minmax_norm_diff(pred, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
                save_plot(pred.squeeze().cpu(), f'{args.log_dir}/pic/{i}/{epoch}_pred.png')
                audio = hifigan(pred)
                save_audio(f'{args.log_dir}/audio/{i}/{epoch}_pred.wav', mel_cfg['sampling_rate'], audio)

                if args.save_ori is True:
                    # save ref
                    # mel_ref = reverse_minmax_norm_diff(mel_ref, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
                    # save_plot(mel_ref.squeeze().cpu(), f'{args.log_dir}/pic/{i}/{epoch}_ref.png')
                    # audio = hifigan(mel_ref)
                    # save_audio(f'{args.log_dir}/audio/{i}/{epoch}_ref.wav', mel_cfg['sampling_rate'], audio)

                    # save source
                    mel = reverse_minmax_norm_diff(mel, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
                    save_plot(mel.squeeze().cpu(), f'{args.log_dir}/pic/{i}/{epoch}_source.png')
                    audio = hifigan(mel)
                    save_audio(f'{args.log_dir}/audio/{i}/{epoch}_source.wav', mel_cfg['sampling_rate'], audio)

                    # save content
                    mean = reverse_minmax_norm_diff(mean, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
                    save_plot(mean.squeeze().cpu(), f'{args.log_dir}/pic/{i}/{epoch}_avg.png')
                    audio = hifigan(mean)
                    save_audio(f'{args.log_dir}/audio/{i}/{epoch}_avg.wav', mel_cfg['sampling_rate'], audio)

            args.save_ori = False
