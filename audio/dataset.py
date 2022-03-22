'''
import os
import random
import torch
import numpy as np
import logging
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from openspeech.data import AUDIO_FEATURE_TRANSFORM_REGISTRY
from openspeech.data.audio.augment import JoiningAugment, NoiseInjector, SpecAugment, TimeStretchAugment
from openspeech.data.audio.load import load_audio

class SpeechTextDataset(Dataset):
    def __init__(self,
                 args,
                 data_path:str,
                 wav_path: list,
                 ):
'''