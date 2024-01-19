from multiprocessing import Process
import os
import numpy as np

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(sr=24000, n_fft=1024, n_mels=100, fmin=0, fmax=12000)

from tqdm import tqdm
import pandas as pd
import pyworld as pw


def get_world_mel(wav_path, sr=24000):
    wav, _ = librosa.load(wav_path, sr=sr)
    wav = (wav * 32767).astype(np.int16)
    wav = (wav / 32767).astype(np.float64)
    # wav = wav.astype(np.float64)
    wav = wav[:(wav.shape[0] // 256) * 256]

    _f0, t = pw.dio(wav, sr, frame_period=256/sr*1000)
    f0 = pw.stonemask(wav, _f0, t, sr)
    sp = pw.cheaptrick(wav, f0, t, sr)
    ap = pw.d4c(wav, f0, t, sr)
    wav_hat = pw.synthesize(f0 * 0, sp, ap, sr, frame_period=256/sr*1000)

    # pyworld output does not pad left
    wav_hat = wav_hat[:len(wav)]
    # wav_hat = wav_hat[256//2: len(wav)+256//2]
    assert len(wav_hat) == len(wav)
    wav = wav_hat.astype(np.float32)
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))

    return log_mel_spectrogram, f0


def chunks(arr, m):
    result = [[] for i in range(m)]
    for i in range(len(arr)):
        result[i%m].append(arr[i])
    return result


def extract_pw(subset, save_f0=False):
    meta = pd.read_csv('../raw_data/meta_fix.csv')
    meta = meta[meta['subset'] == 'train']

    for i in tqdm(subset):
        line = meta.iloc[i]
        audio_dir = '../raw_data/' + line['folder'] + line['subfolder']
        f = line['file_name']

        mel_dir = audio_dir.replace('vocal', 'world').replace('raw_data/', '24k_data/')
        f0_dir = audio_dir.replace('vocal', 'f0').replace('raw_data/', '24k_f0/')

        if os.path.exists(os.path.join(mel_dir, f+'.npy')) is False:
            mel = get_world_mel(os.path.join(audio_dir, f))

            if os.path.exists(mel_dir) is False:
                os.makedirs(mel_dir)
            np.save(os.path.join(mel_dir, f+'.npy'), mel)

            if save_f0 is True:
                if os.path.exists(f0_dir) is False:
                    os.makedirs(f0_dir)
                np.save(os.path.join(f0_dir, f + '.npy'), f0)


if __name__ == '__main__':
    cores = 8
    meta = pd.read_csv('../raw_data/meta_fix.csv')
    meta = meta[meta['subset'] == 'train']

    idx_list = [i for i in range(len(meta))]

    subsets = chunks(idx_list, cores)

    for subset in subsets:
        t = Process(target=extract_pw, args=(subset,))
        t.start()