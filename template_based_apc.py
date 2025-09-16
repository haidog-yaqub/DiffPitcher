import os.path

import numpy as np
import pandas as pd
import torch
import yaml
import librosa
import soundfile as sf
from tqdm import tqdm

from diffusers import DDIMScheduler
from pitch_controller.models.unet import UNetPitcher
from pitch_controller.utils import minmax_norm_diff, reverse_minmax_norm_diff
from pitch_controller.modules.BigVGAN.inference import load_model
from utils import get_mel, get_world_mel, get_f0, f0_to_coarse, show_plot, get_matched_f0, log_f0


@torch.no_grad()
def template_pitcher(source, pitch_ref, model, hifigan, steps=50, shift_semi=0):

    source_mel = get_world_mel(source, sr=sr)

    f0_ref = get_matched_f0(source, pitch_ref, 'world')
    f0_ref = f0_ref * 2 ** (shift_semi / 12)

    target_length = source_mel.shape[-1]
    current_length = f0_ref.shape[-1]

    if current_length != target_length:
        if current_length < target_length:
            f0_ref = np.pad(f0_ref, (0, target_length - current_length), 'constant', constant_values=0)
        else:
            pad_length = current_length - target_length
            min_val = source_mel.min()
            source_mel = np.pad(source_mel, ((0, 0), (0, pad_length)), 'constant', constant_values=min_val)
    
    f0_ref = log_f0(f0_ref, {'f0_bin': 345,
                             'f0_min': librosa.note_to_hz('C2'),
                             'f0_max': librosa.note_to_hz('C#6')})

    source_mel = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
    f0_ref = torch.from_numpy(f0_ref).float().unsqueeze(0).to(device)

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    generator = torch.Generator(device=device).manual_seed(2024)

    noise_scheduler.set_timesteps(steps)
    noise = torch.randn(source_mel.shape, generator=generator, device=device)
    pred = noise
    source_x = minmax_norm_diff(source_mel, vmax=max_mel, vmin=min_mel)

    for t in tqdm(noise_scheduler.timesteps):
        pred = noise_scheduler.scale_model_input(pred, t)
        model_output = model(x=pred, mean=source_x, f0=f0_ref, t=t, ref=None, embed=None)
        pred = noise_scheduler.step(model_output=model_output,
                                    timestep=t,
                                    sample=pred,
                                    eta=1, generator=generator).prev_sample

    pred = reverse_minmax_norm_diff(pred, vmax=max_mel, vmin=min_mel)

    pred_audio = hifigan(pred)
    pred_audio = pred_audio.cpu().squeeze().clamp(-1, 1)

    return pred_audio


if __name__ == '__main__':
    min_mel = np.log(1e-5)
    max_mel = 2.5
    sr = 24000

    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'

    # load diffusion model
    config = yaml.load(open('pitch_controller/config/DiffWorld_24k.yaml'), Loader=yaml.FullLoader)
    mel_cfg = config['logmel']
    ddpm_cfg = config['ddpm']
    unet_cfg = config['unet']
    model = UNetPitcher(**unet_cfg)
    unet_path = 'ckpts/world_fixed_40.pt'

    state_dict = torch.load(unet_path)
    for key in list(state_dict.keys()):
        state_dict[key.replace('_orig_mod.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    if use_gpu:
        model.cuda()
    model.eval()

    #  load vocoder
    hifi_path = 'ckpts/bigvgan_24khz_100band/g_05000000.pt'
    hifigan, cfg = load_model(hifi_path, device=device)
    hifigan.eval()

    pred_audio = template_pitcher('examples/off-key.wav', 'examples/reference.wav', model, hifigan, steps=50, shift_semi=0)
    sf.write('output_template.wav', pred_audio, samplerate=sr)


