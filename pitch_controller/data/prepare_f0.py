# import amfm_decompy.basic_tools as basic
# import amfm_decompy.pYAAPT as pYAAPT
from multiprocessing import Process
import os
import numpy as np
import pandas as pd
import librosa
from librosa.core import load
from tqdm import tqdm


def get_f0(wav_path):
    wav, _ = load(wav_path, sr=24000)
    wav = wav[:(wav.shape[0] // 256) * 256]
    wav = np.pad(wav, 384, mode='reflect')
    f0, _, _ = librosa.pyin(wav, frame_length=1024, hop_length=256, center=False,
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C6'))
    return np.nan_to_num(f0)


def chunks(arr, m):
    result = [[] for i in range(m)]
    for i in range(len(arr)):
        result[i%m].append(arr[i])
    return result


def extract_f0(subset):
    meta = pd.read_csv('../raw_data/meta_fix.csv')
    meta = meta[meta['subset'] == 'train']
    # meta = meta[meta['folder'] == 'VCTK-Corpus/vocal/']

    for i in tqdm(subset):
        line = meta.iloc[i]
        audio_dir = '../raw_data/' + line['folder'] + line['subfolder']
        f = line['file_name']

        f0_dir = audio_dir.replace('vocal', 'f0').replace('raw_data/', '24k_data_f0/')

        try:
            np.load(os.path.join(f0_dir, f+'.npy'))
        except:
            print(line)
            f0 = get_f0(os.path.join(audio_dir, f))
            if os.path.exists(f0_dir) is False:
                os.makedirs(f0_dir, exist_ok=True)
            np.save(os.path.join(f0_dir, f + '.npy'), f0)

        # if os.path.exists(os.path.join(f0_dir, f+'.npy')) is False:
            # f0 = get_yaapt_f0(os.path.join(audio_dir, f))


if __name__ == '__main__':
    cores = 8
    meta = pd.read_csv('../raw_data/meta_fix.csv')
    meta = meta[meta['subset']=='train']
    # meta = meta[meta['folder'] == 'VCTK-Corpus/vocal/']

    idx_list = [i for i in range(len(meta))]

    subsets = chunks(idx_list, cores)

    for subset in subsets:
        t = Process(target=extract_f0, args=(subset,))
        t.start()
