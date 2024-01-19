import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torch
import librosa
from torch.nn import functional as F


def save_curve_plot(pred, midi, gt, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))

    pred[pred == 0] = np.nan
    midi[midi == 0] = np.nan
    gt[gt == 0] = np.nan

    # im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    ax.plot(range(len(pred)), pred, color='tab:green', label='pred')
    ax.plot(range(len(midi)), midi, color='tab:blue', label='midi')
    ax.plot(range(len(gt)), gt, color='grey', label='gt')
    # plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.legend()
    plt.savefig(savepath)
    plt.close()
#
#
# def save_audio(file_path, sampling_rate, audio):
#     audio = np.clip(audio.detach().cpu().squeeze().numpy(), -0.999, 0.999)
#     wavfile.write(file_path, sampling_rate, (audio * 32767).astype("int16"))


def minmax_norm_diff(tensor: torch.Tensor, vmax: float = librosa.note_to_hz('C6'),
                     vmin: float = 0) -> torch.Tensor:
    tensor = torch.clip(tensor, vmin, vmax)
    tensor = 2 * (tensor - vmin) / (vmax - vmin) - 1
    return tensor


def reverse_minmax_norm_diff(tensor: torch.Tensor, vmax: float = librosa.note_to_hz('C6'),
                             vmin: float = 0) -> torch.Tensor:
    tensor = torch.clip(tensor, -1.0, 1.0)
    tensor = (tensor + 1) / 2
    tensor = tensor * (vmax - vmin) + vmin
    return tensor