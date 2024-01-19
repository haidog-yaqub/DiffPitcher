import os
import numpy as np

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(sr=24000, n_fft=1024, n_mels=100, fmin=0, fmax=12000)

from tqdm import tqdm
import pandas as pd

from multiprocessing import Process


# def get_f0(wav_path):
#     wav, _ = load(wav_path, sr=22050)
#     wav = wav[:(wav.shape[0] // 256) * 256]
#     wav = np.pad(wav, 384, mode='reflect')
#     f0, _, _ = librosa.pyin(wav, frame_length=1024, hop_length=256, center=False,
#                             fmin=librosa.note_to_hz('C2'),
#                             fmax=librosa.note_to_hz('C6'))
#     return np.nan_to_num(f0)

def get_mel(wav_path):
    wav, _ = load(wav_path, sr=24000)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram


def chunks(arr, m):
    result = [[] for i in range(m)]
    for i in range(len(arr)):
        result[i%m].append(arr[i])
    return result


def extract_mel(subset):
    meta = pd.read_csv('../raw_data/meta_fix.csv')
    meta = meta[meta['folder'] == 'eval/vocal/']

    for i in tqdm(subset):
        line = meta.iloc[i]
        audio_dir = '../raw_data/' + line['folder'] + line['subfolder']
        f = line['file_name']

        mel_dir = audio_dir.replace('vocal', 'mel').replace('raw_data/', '24k_data/')

        if os.path.exists(os.path.join(mel_dir, f+'.npy')) is False:
            mel = get_mel(os.path.join(audio_dir, f))
            if os.path.exists(mel_dir) is False:
                os.makedirs(mel_dir)
            np.save(os.path.join(mel_dir, f+'.npy'), mel)


if __name__ == '__main__':
    cores = 8

    meta = pd.read_csv('../raw_data/meta_fix.csv')
    meta = meta[meta['folder'] == 'eval/vocal/']

    idx_list = [i for i in range(len(meta))]

    subsets = chunks(idx_list, cores)

    for subset in subsets:
        t = Process(target=extract_mel, args=(subset,))
        t.start()
