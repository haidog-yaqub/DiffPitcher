# from nsf_hifigan.models import load_model
from modules.BigVGAN.inference import load_model
import librosa

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms

import numpy as np
import soundfile as sf


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            center=False,
            power=1.0,
            norm="slaney",
            n_mels=80,
            mel_scale="slaney",
            f_max=8000,
            f_min=0,
        )

    def forward(self, wav):
        wav = F.pad(wav, ((1024 - 256) // 2, (1024 - 256) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


hifigan, cfg = load_model('modules/BigVGAN/ckpt/bigvgan_22khz_80band/g_05000000', device='cuda')
M = LogMelSpectrogram()

source, sr = torchaudio.load("music.mp3")
source = torchaudio.functional.resample(source, sr, 22050)
source = source.unsqueeze(0)
mel = M(source).squeeze(0)

# f0, f0_bin = get_pitch("116_1_pred.wav")
# f0 = torch.tensor(f0).unsqueeze(0)
with torch.no_grad():
    y_hat = hifigan(mel.cuda()).cpu().numpy().squeeze(1)

sf.write('test.wav', y_hat[0], samplerate=22050)