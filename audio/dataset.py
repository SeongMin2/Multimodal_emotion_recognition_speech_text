import os
import random
import torch
import numpy as np
import logging
# from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
from .wav2vec.extract_wav2vec2 import apply_wav2vec

from audio.load import load_wav

logger = logging.getLogger(__name__)


def get_emotion_class(label):
    emotion_class = 0

    if ((label == "Happy") or (label == "Excited")):
        emotion_class = 2
    elif (label == "Angry"):
        emotion_class = 0
    elif (label == "Neutral"):
        emotion_class = 1
    elif (label == "Sad"):
        emotion_class = 3
    return emotion_class

class SpeechTextDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 wav_paths: list,
                 npz_file,
                 transcripts: list,
                 emotions: list,
                 tokenizer,
                 sos_token: int,
                 eos_token: int,
                 sample_rate: int
                 ) -> None:
        super(SpeechTextDataset, self).__init__()
        self.dataset_path = dataset_path
        self.wav_paths = wav_paths
        self.transcripts = transcripts
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.dataset_size = len(self.wav_paths)
        self.transforms = apply_wav2vec
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.sample_rate = sample_rate
        self._load_wav = load_wav

        metaname = os.path.join(self.data_dir, npz_file)
        metadata = np.load(metaname, allow_pickle=True)
        
    
        # 여기 받아오는것은 phone seq까지해야 코드를 짤 수 있을듯
        #for k, element in enumerate(metadata,0):

    def _get_emotion_class(self, label):
        emotion_class = 0

        if ((label == "Happy") or (label == "Excited")):
            emotion_class = 2
        elif (label == "Angry"):
            emotion_class = 0
        elif (label == "Neutral"):
            emotion_class = 1
        elif (label == "Sad"):
            emotion_class = 3
        return emotion_class

    def _parse_wav(self, wav_path: str) -> Tensor:

        signal = self._load_wav(wav_path, sample_rate=self.sample_rate)

        if signal is None:
            logger.warning(f"{wav_path} is not Valid!!")
            return torch.zeros(1000, self.num_mels)

        feature = self.transforms(signal, self.sample_rate)  # 여기에 wav2vec feature 추출하도록

        # normalization인데 선택적으로 할 수 있게 하는게 좋을듯 음...
        '''
        feature -= feature.mean()
        feature /= np.std(feature)

        feature = torch.FloatTensor(feature).transpose(0, 1)
        '''

        return feature

    def _parse_transcript(self, transcript: str) -> list:
        """
        Parses transcript
        Args:
            transcript (str): transcript of audio file
        Returns
            transcript (list): transcript that added <sos> and <eos> tokens
        """
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.sos_token))
        for token in tokens:
            transcript.append(int(token))
        transcript.append(int(self.eos_token))

        return transcript

    def __getitem__(self, idx) -> dict:
        """ Provides paif of audio & transcript """

        ## 여기서 emotion에 대해서 one hot encoder 형태로 바꿔줘야함 아직 안함
        # emotion one hot으로 바꾸는 함수 만들어서 하면 될듯
        
        wav_path = os.path.join(self.dataset_path, self.wav_paths[idx])

        feature = self._parse_wav(wav_path)
        transcript = self._parse_transcript(self.transcripts[idx])
        emotion_label = self._get_emotion_class(self.emotions[idx])

        return {
            'wav2vec_feature' : np.array(feature, dtype=np.float_),
            'transcript' : np.array(transcript, dtype=np.int_),
            'label' : emotion_label
        }

    def __len__(self):
        return len(self.wav_paths)

