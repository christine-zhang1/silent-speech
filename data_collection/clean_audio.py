import sys
import os
import numpy as np

import noisereduce as nr
import soundfile as sf
import librosa

def clean_directory(directory):
    silence, rate = sf.read(os.path.join(directory, '0_audio.flac'))

    audio_file_names = []
    # load audio files in numerical order
    while True:
        i = len(audio_file_names)
        fname = os.path.join(directory, f'{i}_audio.flac')
        if os.path.exists(fname):
            audio_file_names.append(fname)
        else:
            break

    all_audio_file_names = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('_audio.flac')]
    assert len(audio_file_names) == len(all_audio_file_names), 'error discovering audio files'

    all_rmses = []
    for fname in audio_file_names:
        data, rate = sf.read(fname)
        rms = librosa.feature.rms(y=data)[0]
        all_rmses.append(rms)

    silent_cutoff = 0.02
    smoothing_width = 20
    target_rms = 0.2
    clip_to = 0.99

    max_rmses = [np.max(r) for r in all_rmses]
    smoothed_maxes = []
    is_silent = False
    for i in range(len(max_rmses)):
        vs = [max_rmses[j] for j in range(max(0,i-smoothing_width),min(i+1+smoothing_width,len(max_rmses))) if max_rmses[j] > silent_cutoff]
        if len(vs) == 0:
            is_silent = True
            break
        smoothed_maxes.append(np.mean(vs))

    if is_silent:
        print('long run of quiet audio, skipping volume normalization')

    for i, fname in enumerate(audio_file_names):
        data, rate = sf.read(fname)
        print(fname)
        clean = nr.reduce_noise(y=data, sr=rate, y_noise=silence) # cz: what should sr be?
        if np.isnan(clean).any():
            print('skipping', fname)
            continue
        if rate != 22050:
            clean = librosa.resample(y=clean, orig_sr=rate, target_sr=22050)
            rate = 22050
        if not is_silent:
            clean *= target_rms / smoothed_maxes[i]
            max_val = np.abs(clean).max()
            if max_val > clip_to: # this shouldn't happen too often with target_rms of 0.2
                clean = clean / max_val * clip_to

        clean_full_name = fname[:-5] + '_clean.flac'
        sf.write(clean_full_name, clean, rate)

assert len(sys.argv) > 1, 'requires at least 1 argument: the directories to process'
for i in range(1, len(sys.argv)):
    print('cleaning', sys.argv[i])
    clean_directory(sys.argv[i])
