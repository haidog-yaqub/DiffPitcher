import os, json, argparse, yaml
import numpy as np
from tqdm import tqdm
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dataset.diffpitch import DiffPitch
from models.transformer import PitchFormer
from utils import minmax_norm_diff, reverse_minmax_norm_diff, save_curve_plot


parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, default='config/DiffPitch.yaml')

parser.add_argument('-seed', type=int, default=9811)
parser.add_argument('-amp', type=bool, default=False)
parser.add_argument('-compile', type=bool, default=False)

parser.add_argument('-data_dir', type=str, default='data/')
parser.add_argument('-content_dir', type=str, default='world')

parser.add_argument('-train_frames', type=int, default=256)
parser.add_argument('-test_frames', type=int, default=256)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-test_size', type=int, default=1)
parser.add_argument('-num_workers', type=int, default=4)
parser.add_argument('-lr', type=float, default=5e-5)
parser.add_argument('-weight_decay', type=int, default=1e-6)

parser.add_argument('-epochs', type=int, default=1)
parser.add_argument('-save_every', type=int, default=20)
parser.add_argument('-log_step', type=int, default=100)
parser.add_argument('-log_dir', type=str, default='logs_transformer_pitch')
parser.add_argument('-ckpt_dir', type=str, default='ckpt_transformer_pitch')

args = parser.parse_args()
args.save_ori = True
config = yaml.load(open(args.config), Loader=yaml.FullLoader)
mel_cfg = config['logmel']
ddpm_cfg = config['ddpm']
# unet_cfg = config['unet']


def RMSE(gen_f0, gt_f0):
    # Get voiced part
    gt_f0 = gt_f0[0]
    gen_f0 = gen_f0[0]

    nonzero_idxs = np.where((gen_f0 != 0) & (gt_f0 != 0))[0]
    gen_f0_voiced = np.log2(gen_f0[nonzero_idxs])
    gt_f0_voiced = np.log2(gt_f0[nonzero_idxs])
    # log F0 RMSE
    if len(gen_f0_voiced) != 0:
        f0_rmse = np.sqrt(np.mean((gen_f0_voiced - gt_f0_voiced) ** 2))
    else:
        f0_rmse = 0
    return f0_rmse


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

    print('Initializing data loaders...')
    trainset = DiffPitch('data/', 'train', args.train_frames, shift=True)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers,
                              drop_last=True)

    val_set = DiffPitch('data/', 'val', args.test_frames, shift=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    test_set = DiffPitch('data/', 'test', args.test_frames, shift=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    real_set = DiffPitch('data/', 'real', args.test_frames, shift=False)
    read_loader = DataLoader(real_set, batch_size=1, shuffle=False)

    print('Initializing and loading models...')
    model = PitchFormer(mel_cfg['n_mels'], 512).to(args.device)
    ckpt = torch.load('ckpt_transformer_pitch/transformer_pitch_460.pt')
    model.load_state_dict(ckpt)

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
            mel, midi, f0 = batch
            mel = mel.to(args.device)
            midi = midi.to(args.device)
            f0 = f0.to(args.device)

            if args.amp:
                with autocast():
                    f0_pred = model(sp=mel, midi=midi)
                    loss = F.mse_loss(f0_pred, f0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                f0_pred = model(sp=mel, midi=midi)
                loss = F.l1_loss(f0_pred, f0)
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
        torch.save(ckpt, f=f"{args.ckpt_dir}/transformer_pitch_{epoch}.pt")

        print('Inference...\n')
        model.eval()
        with torch.no_grad():
            val_loss = []
            val_rmse = []
            for i, batch in enumerate(val_loader):
                # optimizer.zero_grad()
                mel, midi, f0 = batch
                mel = mel.to(args.device)
                midi = midi.to(args.device)
                f0 = f0.to(args.device)

                f0_pred = model(sp=mel, midi=midi)

                # save pred
                f0_pred[f0_pred < librosa.note_to_hz('C2')] = 0
                f0_pred[f0_pred > librosa.note_to_hz('C6')] = librosa.note_to_hz('C6')

                val_loss.append(F.l1_loss(f0_pred, f0).item())
                val_rmse.append(RMSE(f0_pred.cpu().numpy(), f0.cpu().numpy()))

                if i <= 4:
                    save_path = f'{args.log_dir}/pic/{i}/{epoch}_val.png'
                    if os.path.exists(os.path.dirname(save_path)) is False:
                        os.makedirs(os.path.dirname(save_path))
                    save_curve_plot(f0_pred.cpu().squeeze(), midi.cpu().squeeze(), f0.cpu().squeeze(), save_path)
                # else:
                #     break

            msg = '\nEpoch: [{}][{}]\tLoss: {:.6f}\tRMSE:{:.6f}\n'.\
                format(epoch, args.epochs, np.mean(val_loss), np.mean(val_rmse))
            with open(f'{args.log_dir}/eval_dec.log', 'a') as f:
                f.write(msg)

            test_loss = []
            test_rmse = []
            for i, batch in enumerate(test_loader):
                # optimizer.zero_grad()
                mel, midi, f0 = batch
                mel = mel.to(args.device)
                midi = midi.to(args.device)
                f0 = f0.to(args.device)

                f0_pred = model(sp=mel, midi=midi)

                # save pred
                f0_pred[f0_pred < librosa.note_to_hz('C2')] = 0
                f0_pred[f0_pred > librosa.note_to_hz('C6')] = librosa.note_to_hz('C6')

                test_loss.append(F.l1_loss(f0_pred, f0).item())
                test_rmse.append(RMSE(f0_pred.cpu().numpy(), f0.cpu().numpy()))

                if i <= 4:
                    save_path = f'{args.log_dir}/pic/{i}/{epoch}_test.png'
                    if os.path.exists(os.path.dirname(save_path)) is False:
                        os.makedirs(os.path.dirname(save_path))
                    save_curve_plot(f0_pred.cpu().squeeze(), midi.cpu().squeeze(), f0.cpu().squeeze(), save_path)

            msg = '\nEpoch: [{}][{}]\tLoss: {:.6f}\tRMSE:{:.6f}\n'. \
                format(epoch, args.epochs, np.mean(test_loss), np.mean(test_rmse))
            with open(f'{args.log_dir}/test_dec.log', 'a') as f:
                f.write(msg)

            for i, batch in enumerate(read_loader):
                # optimizer.zero_grad()
                mel, midi, f0 = batch
                mel = mel.to(args.device)
                midi = midi.to(args.device)
                f0 = f0.to(args.device)

                f0_pred = model(sp=mel, midi=midi)
                f0_pred[f0 == 0] = 0

                # save pred
                f0_pred[f0_pred < librosa.note_to_hz('C2')] = 0
                f0_pred[f0_pred > librosa.note_to_hz('C6')] = librosa.note_to_hz('C6')

                save_path = f'{args.log_dir}/pic/{i}/{epoch}_real.png'
                if os.path.exists(os.path.dirname(save_path)) is False:
                    os.makedirs(os.path.dirname(save_path))
                save_curve_plot(f0_pred.cpu().squeeze(), midi.cpu().squeeze(), f0.cpu().squeeze(), save_path)




