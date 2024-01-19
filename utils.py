import numpy as np
import torch
import librosa
from librosa.core import load
import matplotlib.pyplot as plt
import pysptk
import pyworld as pw
from fastdtw import fastdtw
from scipy import spatial

from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(sr=24000, n_fft=1024, n_mels=100, fmin=0, fmax=12000)


def _get_best_mcep_params(fs):
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")


def get_mel(wav_path):
    wav, _ = load(wav_path, sr=24000)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    if mel_spectrogram.shape[-1] % 8 != 0:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 8 - mel_spectrogram.shape[-1] % 8)), 'minimum')

    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram


def get_world_mel(wav_path=None, sr=24000, wav=None):
    if wav_path is not None:
        wav, _ = librosa.load(wav_path, sr=24000)
    wav = (wav * 32767).astype(np.int16)
    wav = (wav / 32767).astype(np.float64)
    # wav = wav.astype(np.float64)
    wav = wav[:(wav.shape[0] // 256) * 256]

    # _f0, t = pw.dio(wav, sr, frame_period=256/sr*1000)
    _f0, t = pw.dio(wav, sr)
    f0 = pw.stonemask(wav, _f0, t, sr)
    sp = pw.cheaptrick(wav, f0, t, sr)
    ap = pw.d4c(wav, f0, t, sr)
    wav_hat = pw.synthesize(f0 * 0, sp, ap, sr)
    # wav_hat = pw.synthesize(f0 * 0, sp, ap, sr, frame_period=256/sr*1000)

    # pyworld output does not pad left
    wav_hat = wav_hat[:len(wav)]
    # wav_hat = wav_hat[256//2: len(wav)+256//2]
    assert len(wav_hat) == len(wav)
    wav = wav_hat.astype(np.float32)
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    if mel_spectrogram.shape[-1] % 8 != 0:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 8 - mel_spectrogram.shape[-1] % 8)), 'minimum')

    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram


def get_f0(wav_path, method='pyin', padding=True):
    if method == 'pyin':
        wav, sr = load(wav_path, sr=24000)
        wav = wav[:(wav.shape[0] // 256) * 256]
        wav = np.pad(wav, 384, mode='reflect')
        f0, _, _ = librosa.pyin(wav, frame_length=1024, hop_length=256, center=False, sr=24000,
                                fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C6'), fill_na=0)
    elif method == 'world':
        wav, sr = librosa.load(wav_path, sr=24000)
        wav = (wav * 32767).astype(np.int16)
        wav = (wav / 32767).astype(np.float64)
        _f0, t = pw.dio(wav, fs=24000, frame_period=256/sr*1000,
                        f0_floor=librosa.note_to_hz('C2'),
                        f0_ceil=librosa.note_to_hz('C6'))
        f0 = pw.stonemask(wav, _f0, t, sr)
        f0 = f0[:-1]

    if padding is True:
        if f0.shape[-1] % 8 !=0:
            f0 = np.pad(f0, ((0, 8-f0.shape[-1] % 8)), 'constant', constant_values=0)

    return f0


def get_mcep(x, n_fft=1024, n_shift=256, sr=24000):
    x, sr = load(x, sr=24000)
    n_frame = (x.shape[0] // 256)
    x = np.pad(x, 384, mode='reflect')
    # n_frame = (len(x) - n_fft) // n_shift + 1
    win = pysptk.sptk.hamming(n_fft)
    mcep_dim, mcep_alpha = _get_best_mcep_params(sr)
    mcep = [pysptk.mcep(x[n_shift * i: n_shift * i + n_fft] * win,
                        mcep_dim, mcep_alpha,
                        eps=1e-6, etype=1,)
            for i in range(n_frame)
            ]
    mcep = np.stack(mcep)
    return mcep


def get_matched_f0(x, y, method='world', n_fft=1024, n_shift=256):
    # f0_x = get_f0(x, method='pyin', padding=False)
    f0_y = get_f0(y, method=method, padding=False)
    # print(f0_y.max())
    # print(f0_y.min())

    mcep_x = get_mcep(x, n_fft=n_fft, n_shift=n_shift)
    mcep_y = get_mcep(y, n_fft=n_fft, n_shift=n_shift)

    _, path = fastdtw(mcep_x, mcep_y, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    # f0_x = gen_mcep[twf[0]]
    nearest = []
    for i in range(len(f0_y)):
        idx = np.argmax(1 * twf[0] == i)
        nearest.append(twf[1][idx])

    f0_y = f0_y[nearest]

    # f0_y = f0_y.astype(np.float32)

    if f0_y.shape[-1] % 8 != 0:
        f0_y = np.pad(f0_y, ((0, 8 - f0_y.shape[-1] % 8)), 'constant', constant_values=0)

    return f0_y


def f0_to_coarse(f0, hparams):

    f0_bin = hparams['f0_bin']
    f0_max = hparams['f0_max']
    f0_min = hparams['f0_min']
    is_torch = isinstance(f0, torch.Tensor)
    # to mel scale
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)

    unvoiced = (f0_mel == 0)

    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1

    f0_mel[unvoiced] = 0

    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 0, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def log_f0(f0, hparams):
    f0_bin = hparams['f0_bin']
    f0_max = hparams['f0_max']
    f0_min = hparams['f0_min']

    f0_mel = np.zeros_like(f0)
    f0_mel[f0 != 0] = 12*np.log2(f0[f0 != 0]/f0_min) + 1
    f0_mel_min = 12*np.log2(f0_min/f0_min) + 1
    f0_mel_max = 12*np.log2(f0_max/f0_min) + 1

    unvoiced = (f0_mel == 0)

    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1

    f0_mel[unvoiced] = 0

    f0_coarse = np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= (f0_bin-1) and f0_coarse.min() >= 0, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def show_plot(tensor):
    tensor = tensor.squeeze().cpu()
    # plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.show()


if __name__ == '__main__':
    mel = get_mel('target.wav')
    f0 = get_f0('target.wav')