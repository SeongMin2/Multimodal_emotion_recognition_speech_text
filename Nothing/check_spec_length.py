import os, sys
import random
import torch
import numpy as np
from audio.wav2vec.extract_wav2vec2 import apply_wav2vec
import parser_helper as helper
#from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

#MODEL_TYPE = "facebook/wav2vec2-large-lv60"


config = helper.get_training_config()
dataset_dir = config.train_dir
npz_file = "train"

metaname = os.path.join(dataset_dir, npz_file + ".npz")
metadata = np.load(metaname, allow_pickle=True)
metadata = metadata['feature']

up_900 = 0
# initialize the dataset
dataset = [None] * len(metadata)
max = 0
max_list = list()
txt_list = list()
for k, element in enumerate(metadata, 0):
    # filter categories
    if str(element[3]) not in config.selected_catg:
        continue

    # load the spectrogram
    spec = np.load(os.path.join(dataset_dir, element[0]))
    if int(spec.shape[0]) > 900:
        up_900 += 1
    if max < int(spec.shape[0]):
        max = int(spec.shape[0])
        max_list.append(max)
        txt_list.append(element[4])
        print(max,":",len(element[4]),":",element[4])
        print(element[0])
        print('')

print(max)
print(max_list)
print(up_900)
print(txt_list)