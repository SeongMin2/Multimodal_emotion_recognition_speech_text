import os, sys
import torch
import numpy as np
import logging
import math
import parser_helper as helper
from torch.utils.data import Dataset
from .wav2vec.extract_wav2vec2 import apply_wav2vec
from text.feature_embedding import extract_features
from transformers import AutoModel, AutoTokenizer

from audio.load import load_wav

logger = logging.getLogger(__name__)

class SpeechTextDataset(Dataset):
    def __init__(self,
                 config,
                 mode: str,
                 dataset_dir: str, # 이거는 그 spectrum이랑, npz데이터 있는 train_dataset path
                 wav_dir: str, # 이거는 원래 날 것의 wav파일들의 path -> get_item할 때 마다 wav2vec2 하려고 ㅎ
                 npz_file,
                 sample_rate: int
                 ) -> None:
        super(SpeechTextDataset, self).__init__()
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.wav_dir = wav_dir
        self.len_crop = config.len_crop
        self.speech_input = config.speech_input
        #self.txt_feat_model = config.pretrained_text_model
        self.transforms = apply_wav2vec # 일단 default로 wav2vec2.0 transform으로 해둠 (일단은)
        self.sample_rate = sample_rate
        self._load_wav = load_wav
        self.max_token_len = config.max_token_len

        assert self.mode == "train" or "test", "mode should be 'train' or 'test'."

        metaname = os.path.join(self.dataset_dir, npz_file + ".npz")
        metadata = np.load(metaname, allow_pickle=True)
        metadata = metadata['feature']

        # initialize the dataset
        dataset = [None] * len(metadata)

        for k, element in enumerate(metadata, 0):
            # filter categories
            if str(element[3]) not in config.selected_catg:
                continue

            element[3] = self._get_emotion_class(element[3]) # element[3] is emotion class

            # load the spectrogram
            spec = np.load(os.path.join(self.dataset_dir, element[0].replace("\\","/")))

            # check if audio and features have the same shape
            self.check_audio_and_feat_shape(audio=spec, element=element)

            dataset[k] = self.get_dataset_sample(spec=spec, element=element,config=config)

        while None in dataset:
            dataset.remove(None)


        self.dataset = list(dataset) # 여기에서의 길이를 조절하여서 실험하기
        self.num_tokens = len(self.dataset)

        #self.tokenizer = AutoTokenizer.from_pretrained(self.txt_feat_model)
        #self.text_model = AutoModel.from_pretrained(self.txt_feat_model)

        #print("Num utterances: ", self.num_tokens)  # number of utterances
        #print("Finished loading the dataset...")

        helper.logger("info", "[INFO] Dataset info - mode: {} / dir: {}".format(self.mode, self.dataset_dir))
        helper.logger("info", "[INFO] Num utterances: {}".format(self.num_tokens))
        helper.logger("info", "[INFO] Finished loading the dataset...")
        
        # log 나중에 할까나

    def check_audio_and_feat_shape(self, audio, element):
        """ Check if the audio shape matches the features shape """
        if not (audio.shape[0] == element[2].shape[0]):
            print("[ERROR] Audio and main phone sequence do not have the same size")
            sys.exit(1)

    def get_dataset_sample(self, spec, element, config):
        """ Return a dataset sample """
        data_sample = [element[0], spec, element[-2], element[2], element[3], element[4], element[-3]]

        if config.speech_input == "wav2vec":
            self.transforms = apply_wav2vec

        elif config.speech_input == "spec":
            pass # 일단 pass 나중에는 spec 변환하는 함수로 바꾸기
        
        return data_sample

    def select_data_sample_to_return(self, gt_config, features):
        """ Determines how to return the data sample """

        if gt_config["speech_input"] == "wav2vec":
            return {
                "spec": features["spec"],
                "spk_emb": features["spk_emb"],
                "phones": features["phones"],
                "txt_feat" : features["txt_feat"],
                "attn_mask_ids" : features["attn_mask_ids"],
                "emotion_lb": features["emotion_lb"],
                "wav2vec_feat": features["wav2vec_feat"]
            }
            # return features["spec"], features["spk_emb"], features["phones"], features["emotion_lb"], features["text"], features["wav2vec_feat"]
            # spectrum, emb_org, phone, emotion_label, wav2vec_feature
        elif gt_config["speech_input"] == "spec":
            return {
                "spec": features["spec"],
                "spk_emb": features["spk_emb"],
                "phones": features["phones"],
                "txt_feat" : features["txt_feat"],
                "emotion_lb": features["emotion_lb"],
                "attn_mask_ids": features["attn_mask_ids"]
            }

    def no_cut_or_pad_features(self, features, as_list):
        """ Return features without cutting or padding them """

        if as_list:
            features["spec"] = [features["spec"]]
            features["phones"] = [features["phones"]]
            features["wav2vec_feat"] = [features["wav2vec_feat"]]
            return features
        else:
            return features

    def zero_pad(self, array, len_pad):
        """ Zero pads a 2d array with zeros to the right """
        return np.pad(array, ((0, len_pad), (0, 0)), "constant")

    def zero_pad_feats(self, features, gt_config, as_list=False):
        """ Zero pads the features that are too short"""

        wav2vec_feat = None
        len_pad = gt_config["len_crop"] - features["spec"].shape[0]
        features['spec'] = self.zero_pad(features["spec"], len_pad)
        features["phones"] = self.zero_pad(features["phones"], len_pad)


        if gt_config["speech_input"] == "wav2vec":
            features["wav2vec_feat"] = np.pad(features["wav2vec_feat"], ((0, 0), (0, len_pad), (0, 0)), "constant")

        if as_list:
            features['spec'] = [features['spec']]
            features['phones'] = [features['phones']]
            features['wav2vec_feat'] = [features['wav2vec_feat']]

        return features


    def feats_crop(self, features, left, gt_config):
        """ Crops the features from a starting point to the left """
        features["spec"] = features["spec"][left:left + gt_config["len_crop"], :]
        features["phones"] = features["phones"][left:left + gt_config["len_crop"], :]
        if gt_config["speech_input"] == "wav2vec":
            features["wav2vec_feat"] = features["wav2vec_feat"][:, left:left + gt_config["len_crop"], :]

        return features

    def random_feats_crop(self, features, gt_config):
        """ Crops the features randomly """
        # randomly crop the utterance
        left = np.random.randint(features["spec"].shape[0] - gt_config["len_crop"])
        return self.feats_crop(features, left, gt_config)

    def crop_utt_segments(self, features, gt_config):
        """ For all the features, crop all the segments in an utterance
            and return them as lists """
        num_segments = math.floor(features["spec"].shape[0] / gt_config["len_crop"]) #  원래 spec 크기 / 96
        uttr = []
        content_emb = []
        wav2vec_feat = []
        # crop and append the segments
        for seg_i in range(num_segments):
            uttr.append(features["spec"][seg_i * gt_config["len_crop"]:(seg_i + 1) * gt_config["len_crop"], :])
            content_emb.append(features["phones"][seg_i * gt_config["len_crop"]:(seg_i + 1) * gt_config["len_crop"], :])
            if gt_config["speech_input"] == "wav2vec":
                wav2vec_feat.append(
                    features["wav2vec_feat"][:, seg_i * gt_config["len_crop"]:(seg_i + 1) * gt_config["len_crop"],:])
        # check if there is a last segment that needs padding
        if ((features["spec"].shape[0] - num_segments * gt_config["len_crop"]) > 1):
            len_pad = gt_config["len_crop"] - (features["spec"].shape[0] - gt_config["len_crop"] * num_segments)
            uttr.append(
                np.pad(features["spec"][num_segments * gt_config["len_crop"]:, :], ((0, len_pad), (0, 0)), "constant"))
            content_emb.append(
                np.pad(features["phones"][num_segments * gt_config["len_crop"]:, :], ((0, len_pad), (0, 0)),
                       "constant"))
            if gt_config["speech_input"] == "wav2vec":
                wav2vec_feat.append(np.pad(features["wav2vec_feat"][:, num_segments * gt_config["len_crop"]:, :],
                                           ((0, 0), (0, len_pad), (0, 0)), "constant"))

        features["spec"] = uttr
        features["phones"] = content_emb
        features["wav2vec_feat"] = wav2vec_feat

        return features

    def _get_emotion_class(self, label):
        emotion_class = 0

        if ((label == "hap") or (label == "exc")):
            emotion_class = 2
        elif (label == "ang"):
            emotion_class = 0
        elif (label == "neu"):
            emotion_class = 1
        elif (label == "sad"):
            emotion_class = 3
        return emotion_class

    def _parse_wav(self, wav_path: str):

        signal = self._load_wav(wav_path, sample_rate=self.sample_rate)

        if signal is None:
            logger.warning(f"{wav_path} is not Valid!!")
            return torch.zeros(1000, self.num_mels)

        feature = self.transforms(signal, self.sample_rate)  # 여기에 wav2vec feature 추출하도록

        feature = np.array(feature) # list를 numpy로 한 번 바꿔주고 이를 tensor로 바꿈

        # normalization인데 선택적으로 할 수 있게 하는게 좋을듯 음...
        '''
        feature -= feature.mean()
        feature /= np.std(feature)

        feature = torch.FloatTensor(feature).transpose(0, 1)
        '''

        return feature

    def _parse_transcript(self, transcript: str):

        feature, mask_s_idx = extract_features(transcript, self.max_token_len) #, self.tokenizer, self.text_model)

        return feature, mask_s_idx

    def __getitem__(self, idx): # -> dict:
        """ Provides paif of audio & transcript """

        gt_config={} # GetiTem (gt)
        gt_config["len_crop"] = self.len_crop
        gt_config["speech_input"] = self.speech_input

        data_sample = self.dataset[idx]

        # get datasample basic features
        file_name, spec, spk_emb, phones, emotion_class, txt = data_sample[0:6]

        npy_path = str(file_name)
        wav_path = str(npy_path[:-4] + ".wav")
        wav_path = os.path.join(self.wav_dir, wav_path)

        wav2vec_feat = self._parse_wav(wav_path.replace("\\", "/")) # wav2vec feature results

        txt_feat, txt_mask_s_idx = self._parse_transcript(txt)
        # text preprocessing
        txt_feat = txt_feat[1:-1] # It discard start text token
        txt_mask_s_idx -= 1 # The reason why this code is reasonable is that end token has to be removed
        # 여러 tokenizer에 따라서 padding 진행하긴하지만 애초에 tokenizer에서 max_length박아서 padding해서 일단 안한다.

        txt_feat = torch.from_numpy(txt_feat)

        spch_mask_s_idx = self.len_crop # padding 안하는 경우
        pad_len = self.len_crop - spec.shape[0]
        if pad_len > 0:
            spch_mask_s_idx = self.len_crop - pad_len


        features = {}
        features["spec"] = spec
        features["phones"] = phones
        features["spk_emb"] = spk_emb
        features["wav2vec_feat"] = wav2vec_feat
        features["text"] = txt
        features['txt_feat'] = txt_feat
        features['attn_mask_ids'] = np.array([txt_mask_s_idx, spch_mask_s_idx]) # start_idx
        features["emotion_lb"] = emotion_class

        # 이 padding 해주는 부분에 대해서 debugging으로 확인하기 위에서는 np.pad하고 ()로 묶어서 return 하던데 
        # 이거 확인해야함
        if spec.shape[0] < self.len_crop:
            if self.mode == "train":
                features = self.zero_pad_feats(features=features, gt_config=gt_config)
            elif self.mode == "test":
                features = self.zero_pad_feats(features=features, gt_config=gt_config, as_list=True)
        # if the utterance is too long (crop)
        elif spec.shape[0] > self.len_crop:
            if self.mode == "train":
                # randomly crop the utterance
                features = self.random_feats_crop(features, gt_config)
            elif self.mode == "test":
                features = self.crop_utt_segments(features,gt_config)
        # if the utterance has the exact crop size
        else:
            if self.mode == "test":
                features = self.no_cut_or_pad_features(features=features, as_list=True)


        return (self.select_data_sample_to_return(gt_config, features))
        # return 할 때는 무조건 최소 numpy로 해야 함 list로 하면 dataloader에서 불러올 때 기괴하게 리스트 안에 tensor 있고 그럼

    def __len__(self):
        return len(self.dataset)


# 가서 해야할 것
# fold 단위로 npz 파일 저장시키기 위에 이거 직접 돌려보기
# crop pad부분에 대해서 유의해서 보기