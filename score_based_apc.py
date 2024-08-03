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
from pitch_predictor.models.transformer import PitchFormer
import pretty_midi


def prepare_midi_wav(wav_id, midi_id, sr=24000):
    midi = pretty_midi.PrettyMIDI(midi_id)
    roll = midi.get_piano_roll()
    roll = np.pad(roll, ((0, 0), (0, 1000)), constant_values=0)
    roll[roll > 0] = 100

    onset = midi.get_onsets()
    before_onset = list(np.round(onset * 100 - 1).astype(int))
    roll[:, before_onset] = 0

    wav, sr = librosa.load(wav_id, sr=sr)

    start = 0
    end = round(100 * len(wav) / sr) / 100
    # save audio
    wav_seg = wav[round(start * sr):round(end * sr)]
    cur_roll = roll[:, round(100 * start):round(100 * end)]
    return wav_seg, cur_roll


def algin_mapping(content, target_len):
    # align content with mel
    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    temp = torch.arange(src_len+1) * target_len / src_len

    for i in range(target_len):
        cur_idx = torch.argmin(torch.abs(temp-i))
        target[:, i] = content[:, cur_idx]
    return target


def midi_to_hz(midi):
    idx = torch.zeros(midi.shape[-1])
    for frame in range(midi.shape[-1]):
        midi_frame = midi[:, frame]
        non_zero = midi_frame.nonzero()
        if len(non_zero) != 0:
            hz = librosa.midi_to_hz(non_zero[0])
            idx[frame] = torch.tensor(hz)
    return idx


@torch.no_grad()
def score_pitcher(source, pitch_ref, model, hifigan, pitcher, steps=50, shift_semi=0, mask_with_source=False):
    wav, midi = prepare_midi_wav(source, pitch_ref, sr=sr)

    source_mel = get_world_mel(None, sr=sr, wav=wav)

    midi = torch.tensor(midi, dtype=torch.float32)
    midi = algin_mapping(midi, source_mel.shape[-1])
    midi = midi_to_hz(midi)

    f0_ori = np.nan_to_num(get_f0(source))

    source_mel = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
    f0_ori = torch.from_numpy(f0_ori).float().unsqueeze(0).to(device)
    midi = midi.unsqueeze(0).to(device)

    f0_pred = pitcher(sp=source_mel, midi=midi)
    if mask_with_source:
        # mask unvoiced frames based on original pitch estimation
        f0_pred[f0_ori == 0] = 0
    f0_pred = f0_pred.cpu().numpy()[0]
    # limit range
    f0_pred[f0_pred < librosa.note_to_hz('C2')] = 0
    f0_pred[f0_pred > librosa.note_to_hz('C6')] = librosa.note_to_hz('C6')

    f0_pred = f0_pred * (2 ** (shift_semi / 12))

    f0_pred = log_f0(f0_pred, {'f0_bin': 345,
                               'f0_min': librosa.note_to_hz('C2'),
                               'f0_max': librosa.note_to_hz('C#6')})
    f0_pred = torch.from_numpy(f0_pred).float().unsqueeze(0).to(device)

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    generator = torch.Generator(device=device).manual_seed(2024)

    noise_scheduler.set_timesteps(steps)
    noise = torch.randn(source_mel.shape, generator=generator, device=device)
    pred = noise
    source_x = minmax_norm_diff(source_mel, vmax=max_mel, vmin=min_mel)

    for t in tqdm(noise_scheduler.timesteps):
        pred = noise_scheduler.scale_model_input(pred, t)
        model_output = model(x=pred, mean=source_x, f0=f0_pred, t=t, ref=None, embed=None)
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

    # load pitch predictor
    pitcher = PitchFormer(100, 512).to(device)
    ckpt = torch.load('ckpts/ckpt_transformer_pitch/transformer_pitch_360.pt')
    pitcher.load_state_dict(ckpt)
    pitcher.eval()

    pred_audio = score_pitcher('examples/score_vocal.wav', 'examples/score_midi.midi', model, hifigan, pitcher, steps=50)
    sf.write('output_score.wav', pred_audio, samplerate=sr)




