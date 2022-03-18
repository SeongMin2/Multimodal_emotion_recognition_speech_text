import os
import sys
import numpy as np
import pandas as pd
import tqdm
import torch
import librosa

# WAVE_PATH = '../IEMOCAP_full_release/Session{}/sentences/wav'


labels_df = pd.read_csv('../extraction/df_iemocap_sess1.csv')
SR = 16000
audio_list = []

for sess in [1]:
    wav_file_path = '../IEMOCAP_full_release/Session{}/sentences/wav'.format(sess)
    audio_list = os.listdir(wav_file_path)

    for audio in audio_list:
        wav_files = os.listdir(wav_file_path + audio)

        for wav_file in wav_files:
            vector, _sr = librosa.load(wav_file_path+audio + '/' + wav_file, sr = SR)
            audio_list.append(vector)




sr = 44100
vector, _sr = librosa.load('../IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav',sr=sr)
