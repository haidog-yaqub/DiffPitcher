import os
import random
import numpy as np
import pandas as pd
import librosa

import torch
import torchaudio
from torch.utils.data import Dataset


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


# training "average voice" encoder
class DiffPitch(Dataset):
    def __init__(self, data_dir, subset, frames, content='world', shift=True, log_scale=False):
        meta = pd.read_csv(data_dir + 'meta.csv')
        self.data_dir = data_dir
        self.meta = meta[meta['subset'] == subset]
        self.frames = frames
        self.content = content
        self.shift = shift
        self.log_scale = log_scale

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        folder = row['folder']
        subfolder = row['subfolder']
        file_id = row['file_name']
        folder = os.path.join(self.data_dir, folder)
        folder = os.path.join(folder, str(subfolder))
        folder = os.path.join(folder, 'vocal')
        folder = os.path.join(folder, file_id)

        content_folder = folder.replace('vocal', self.content).replace('.wav', '.npy')
        content = torch.tensor(np.load(content_folder), dtype=torch.float32)
        # print(content.shape)

        midi_folder = folder.replace('vocal', 'roll_align').replace('.wav', '.npy')
        midi = torch.tensor(np.load(midi_folder), dtype=torch.float32)
        # print(midi.shape)
        # midi = algin_mapping(midi, content.shape[-1])

        f0_folder = folder.replace('vocal', 'f0').replace('.wav', '.npy')
        f0 = torch.tensor(np.load(f0_folder), dtype=torch.float32)

        max_start = max(content.shape[-1] - self.frames, 0)
        start = random.choice(range(max_start)) if max_start > 0 else 0
        end = min(int(start + self.frames), content.shape[-1])

        out_content = torch.ones((content.shape[0], self.frames)) * np.log(1e-5)
        out_midi = torch.zeros(self.frames)
        out_f0 = torch.zeros(self.frames)

        out_content[:, :end-start] = content[:, start:end]
        out_midi[:end-start] = midi[start:end]
        out_f0[:end-start] = f0[start:end]

        # out_midi = midi_to_hz(out_midi)

        if self.shift is True:
            shift = np.random.choice(25, 1)[0]
            shift = shift - 12

            # midi[midi != 0] += shift
            out_midi = out_midi*(2**(shift/12))
            out_f0 = out_f0*(2**(shift/12))

        if self.log_scale:
            out_midi = 1127 * np.log(1 + out_midi / 700)
            out_f0 = 1127 * np.log(1 + out_f0 / 700)

        return out_content, out_midi, out_f0

    def __len__(self):
        return len(self.meta)