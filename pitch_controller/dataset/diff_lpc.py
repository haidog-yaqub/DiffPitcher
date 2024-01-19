import os
import random
import numpy as np
import torch
import tgt
import pandas as pd

from torch.utils.data import Dataset
import librosa


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


# training "average voice" encoder
class VCDecLPCDataset(Dataset):
    def __init__(self, data_dir, subset, content_dir='lpc_mel_512', extract_emb=False,
                 f0_type='bins'):
        self.path = data_dir
        meta = pd.read_csv(data_dir + 'meta_fix.csv')
        self.meta = meta[meta['subset'] == subset]
        self.content_dir = content_dir
        self.extract_emb = extract_emb
        self.f0_type = f0_type

    def get_vc_data(self, audio_path, mel_id):
        mel_dir = audio_path.replace('vocal', 'mel')
        embed_dir = audio_path.replace('vocal', 'embed')
        pitch_dir = audio_path.replace('vocal', 'f0')
        content_dir = audio_path.replace('vocal', self.content_dir)

        mel = os.path.join(mel_dir, mel_id + '.npy')
        embed = os.path.join(embed_dir, mel_id + '.npy')
        pitch = os.path.join(pitch_dir, mel_id + '.npy')
        content = os.path.join(content_dir, mel_id + '.npy')

        mel = np.load(mel)
        if self.extract_emb:
            embed = np.load(embed)
        else:
            embed = np.zeros(1)

        pitch = np.load(pitch)
        content = np.load(content)

        pitch = np.nan_to_num(pitch)
        if self.f0_type == 'bins':
            pitch = f0_to_coarse(pitch, {'f0_bin': 256,
                                         'f0_min': librosa.note_to_hz('C2'),
                                         'f0_max': librosa.note_to_hz('C6')})
        elif self.f0_type == 'log':
            pitch = log_f0(pitch, {'f0_bin': 345,
                                   'f0_min': librosa.note_to_hz('C2'),
                                   'f0_max': librosa.note_to_hz('C#6')})

        mel = torch.from_numpy(mel).float()
        embed = torch.from_numpy(embed).float()
        pitch = torch.from_numpy(pitch).float()
        content = torch.from_numpy(content).float()

        return (mel, embed, pitch, content)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        mel_id = row['file_name']
        audio_path = self.path + row['folder'] + row['subfolder']
        mel, embed, pitch, content = self.get_vc_data(audio_path, mel_id)
        item = {'mel': mel, 'embed': embed, 'f0': pitch, 'content': content}
        return item

    def __len__(self):
        return len(self.meta)


class VCDecLPCBatchCollate(object):
    def __init__(self, train_frames, eps=1e-5):
        self.train_frames = train_frames
        self.eps = eps

    def __call__(self, batch):
        train_frames = self.train_frames
        eps = self.eps

        B = len(batch)
        embed = torch.stack([item['embed'] for item in batch], 0)

        n_mels = batch[0]['mel'].shape[0]
        content_dim = batch[0]['content'].shape[0]

        # min value of log-mel spectrogram is np.log(eps) == padding zero in time domain
        mels1 = torch.ones((B, n_mels, train_frames), dtype=torch.float32) * np.log(eps)
        mels2 = torch.ones((B, n_mels, train_frames), dtype=torch.float32) * np.log(eps)

        # ! need to deal with empty frames here
        contents1 = torch.ones((B, content_dim, train_frames), dtype=torch.float32) * np.log(eps)

        f0s1 = torch.zeros((B, train_frames), dtype=torch.float32)
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]

        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            f0 = item['f0']
            content = item['content']

            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames

            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            f0s1[i, :mel_length] = f0[starts1[i]:starts1[i] + mel_length]
            contents1[i, :, :mel_length] = content[:, starts1[i]:starts1[i] + mel_length]

            mels2[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_lengths.append(mel_length)

        mel_lengths = torch.LongTensor(mel_lengths)

        return {'mel1': mels1, 'mel2': mels2, 'mel_lengths': mel_lengths,
                'embed': embed,
                'f0_1': f0s1,
                'content1': contents1}


class VCDecLPCTest(Dataset):
    def __init__(self, data_dir, subset='test', eps=1e-5, test_frames=256, content_dir='lpc_mel_512', extract_emb=False,
                 f0_type='bins'):
        self.path = data_dir
        meta = pd.read_csv(data_dir + 'meta_test.csv')
        self.meta = meta[meta['subset'] == subset]
        self.content_dir = content_dir
        self.extract_emb = extract_emb
        self.eps = eps
        self.test_frames = test_frames
        self.f0_type = f0_type

    def get_vc_data(self, audio_path, mel_id, pitch_shift):
        mel_dir = audio_path.replace('vocal', 'mel')
        embed_dir = audio_path.replace('vocal', 'embed')
        pitch_dir = audio_path.replace('vocal', 'f0')
        content_dir = audio_path.replace('vocal', self.content_dir)

        mel = os.path.join(mel_dir, mel_id + '.npy')
        embed = os.path.join(embed_dir, mel_id + '.npy')
        pitch = os.path.join(pitch_dir, mel_id + '.npy')
        content = os.path.join(content_dir, mel_id + '.npy')

        mel = np.load(mel)
        if self.extract_emb:
            embed = np.load(embed)
        else:
            embed = np.zeros(1)

        pitch = np.load(pitch)
        content = np.load(content)

        pitch = np.nan_to_num(pitch)
        pitch = pitch*pitch_shift

        if self.f0_type == 'bins':
            pitch = f0_to_coarse(pitch, {'f0_bin': 256,
                                         'f0_min': librosa.note_to_hz('C2'),
                                         'f0_max': librosa.note_to_hz('C6')})
        elif self.f0_type == 'log':
            pitch = log_f0(pitch, {'f0_bin': 345,
                                   'f0_min': librosa.note_to_hz('C2'),
                                   'f0_max': librosa.note_to_hz('C#6')})

        mel = torch.from_numpy(mel).float()
        embed = torch.from_numpy(embed).float()
        pitch = torch.from_numpy(pitch).float()
        content = torch.from_numpy(content).float()

        return (mel, embed, pitch, content)

    def __getitem__(self, index):
        row = self.meta.iloc[index]

        mel_id = row['content_file_name']
        audio_path = self.path + row['content_folder'] + row['content_subfolder']
        pitch_shift = row['pitch_shift']
        mel1, _, f0, content = self.get_vc_data(audio_path, mel_id, pitch_shift)

        mel_id = row['timbre_file_name']
        audio_path = self.path + row['timbre_folder'] + row['timbre_subfolder']
        mel2, embed, _, _ = self.get_vc_data(audio_path, mel_id, pitch_shift)

        n_mels = mel1.shape[0]
        content_dim = content.shape[0]

        mels1 = torch.ones((n_mels, self.test_frames), dtype=torch.float32) * np.log(self.eps)
        mels2 = torch.ones((n_mels, self.test_frames), dtype=torch.float32) * np.log(self.eps)
        lpcs1 = torch.ones((content_dim, self.test_frames), dtype=torch.float32) * np.log(self.eps)

        f0s1 = torch.zeros(self.test_frames, dtype=torch.float32)

        if mel1.shape[-1] < self.test_frames:
            mel_length = mel1.shape[-1]
        else:
            mel_length = self.test_frames
        mels1[:, :mel_length] = mel1[:, :mel_length]
        f0s1[:mel_length] = f0[:mel_length]
        lpcs1[:, :mel_length] = content[:, :mel_length]

        if mel2.shape[-1] < self.test_frames:
            mel_length = mel2.shape[-1]
        else:
            mel_length = self.test_frames
        mels2[:, :mel_length] = mel2[:, :mel_length]

        return {'mel1': mels1, 'mel2': mels2, 'embed': embed, 'f0_1': f0s1, 'content1': lpcs1}

    def __len__(self):
        return len(self.meta)


if __name__ == '__main__':
    f0 = np.array([110.0, 220.0, librosa.note_to_hz('C2'), 0, librosa.note_to_hz('E3'), librosa.note_to_hz('C6')])
    # 50 midi notes = (50-1)
    pitch = log_f0(f0, {'f0_bin': 345,
                        'f0_min': librosa.note_to_hz('C2'),
                        'f0_max': librosa.note_to_hz('C#6')})