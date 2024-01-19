import os.path
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import textgrid
import pretty_midi
import music21
import librosa
import soundfile as sf


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0, bpm=120):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, )

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs * bpm / 120
        # time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)

    beats = np.array([0, int(pm.get_end_time()+1)])
    pm.adjust_times(beats, beats * 120 / bpm)
    # print(beats)
    return pm


f = open('CSD/English/metadata.json', encoding="utf8")
meta = json.load(f)
folder = 'CSD/English/'
for wav in tqdm(os.listdir(folder+'wav')):
    song_id = wav.replace('.wav', '')
    midi_id = wav.replace('.wav', '.mid')
    roll_id = wav.replace('.wav', '.npy')

    wav, sr = librosa.load(folder+'wav/'+wav)
    midi = pretty_midi.PrettyMIDI(folder+'mid/'+midi_id)
    roll = midi.get_piano_roll()

    bpm = meta[song_id]['tempo']

    for i in range(int(roll.shape[1])//1000):
        # print(i)
        start = i*10
        end = (i+1)*10

        wav_seg = wav[round(start * sr):round(end * sr)]

        os.makedirs('CSD_segements/'+song_id+'/vocal/', exist_ok=True)
        os.makedirs('CSD_segements/' + song_id + '/roll/', exist_ok=True)
        os.makedirs('CSD_segements/' + song_id + '/midi/', exist_ok=True)

        sf.write('CSD_segements/'+song_id+'/vocal/'+str(i)+'.wav', wav_seg, samplerate=sr)

        cur_roll = roll[:, round(100*start):round(100*end)]

        if round((end-start)*100) != cur_roll.shape[1]:
            print(sentence)
            print(song_id)
            print((end-start)*100)
            print(cur_roll.shape)

        # save npy rolls
        np.save('CSD_segements/'+song_id+'/roll/'+str(i)+'.npy', cur_roll)

        # save midi files
        cur_midi = piano_roll_to_pretty_midi(cur_roll, fs=100, bpm=bpm)
        # cur_midi.write('cache/'+song_id+str(num)+'.midi')
        cur_midi.write('CSD_segements/'+song_id+'/midi/'+str(i)+'.midi')
        # fctr = bpm/120
        # score = music21.converter.Converter()
        # score.parseFile('cache/'+song_id+str(num)+'.midi')
        # newscore = score.stream.augmentOrDiminish(fctr)
        # newscore.write('midi', 'segements/'+song_id+'/midi/'+str(num)+'.midi')