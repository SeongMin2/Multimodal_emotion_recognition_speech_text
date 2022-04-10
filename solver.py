import sys
from os.path import dirname, join, abspath
from model.multimodal import Multimodal
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import numpy as np
import csv
from tqdm import tqdm
from pathlib import Path

def get_checkpoint_path(checkpoint_path_str):
    """ Return the checkpoint path if it exists """
    checkpoint_path = Path(checkpoint_path_str)

    if not checkpoint_path.exists():
        return None
    else:
        return checkpoint_path_str

class Solver(object):
    def __init__(self, config, train_loader, test_loader, train_eval_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_eval_loader = train_eval_loader

        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




        self.pretrained_check = get_checkpoint_path(config.pretrained_model)

        self.build_model()

    def build_model(self):
        self.model = Multimodal(self.config)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        if not (self.pretrained_check == None):  # 만약 pretrained 된 모델 파일이 존재한다면,
            #with open(self.test_out_file, "a") as txt_f:
            #    txt_f.write("Loading the weights at " + self.checkpoint_path + "\n")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def data_to_device(self, vars, state):
        # 이거 뭔가 다른 경우에도 쓸 수 있도록 특졍 변수에 결과를 append 하는 식으로 해서 return 하는식이 더 좋을듯
        spec, spk_emb, phones, txt_feat, emotion_lb = vars

        spec = spec.to(self.device)
        spec = spec.type(torch.cuda.FloatTensor) # Shoud i..?

        spk_emb = spk_emb.to(self.device)
        spk_emb = spk_emb.type(torch.cuda.FloatTensor)

        phones = phones.to(self.device)
        phones = phones.type(torch.cuda.FloatTensor)

        txt_feat = txt_feat.to(self.device)
        txt_feat = txt_feat.type(torch.cuda.FloatTensor)

        if state == "train":
            emotion_lb = emotion_lb.to(self.device)
            emotion_lb = emotion_lb.type(torch.cuda.LongTensor)
            # 왜 IntTensor로 안하는것이지
        elif state == "test":
            emotion_lb = emotion_lb.numpy()[0]

        return spec, spk_emb, phones, txt_feat, emotion_lb

    def calc_UA(self, logit, ground_truth):
        max_vals, max_indices = torch.max(logit, 1)
        ua = (max_indices == ground_truth).sum().data.cpu().numpy()/max_indices.size()[0]

        return ua

    def calc_WA(logit, ground_truth):
        n_tp = [0 for i in range(4)]
        n_tp_fn = [0 for k in range(4)]

        max_vals, max_indices = torch.max(logit, 1)
        tmp_correct = (max_indices.numpy() == ground_truth.numpy())
        for i, idx in enumerate(max_indices.numpy()):
            n_tp[idx] += tmp_correct[idx]
            n_tp_fn[ground_truth[i]] += 1

        return n_tp, n_tp_fn


    def train(self):
        data_loader = self.train_loader
        keys = ['train_loss', 'test_loss']

        for epoch in range(self.config.epochs):
            # To calculate Weighted Accuracy
            train_tp = []
            train_tp_fn = []
            test_tp = []
            test_tp_fn = []

            # To calculate Unweighted Accuracy
            train_ua = 0.0
            test_ua = 0.0

            self.model.train()

            for batch_id, batch in enumerate(tqdm(data_loader)):
                self.optimizer.zero_grad()

                if self.config.input_speech == "wav2vec":
                    spec, spk_emb, phones, txt_feat, emotion_lb, wav2vec_feat = batch.values()
                    wav2vec_feat = wav2vec_feat.to(self.device)
                    wav2vec_feat = wav2vec_feat.type(torch.cuda.FloatTensor)
                elif self.config.input_speech == "spec":
                    spec, spk_emb, phones, txt_feat, emotion_lb = batch.values()

                tmp_vars = self.data_to_device(vars=[spec, spk_emb, phones, txt_feat, emotion_lb],
                                               state="train")
                spec, spk_emb, phones, txt_feat, emotion_lb = tmp_vars

                spec_out, post_spec_out, emotion_logits = self.model(spec, spk_emb, spk_emb, phones, wav2vec_feat, txt_feat)

                spec_loss = F.mse_loss(spec, spec_out)
                post_spec_loss = F.mse_loss(spec, post_spec_out)

                emotion_loss = self.loss_fn(emotion_logits, emotion_lb)

                total_loss = spec_loss + post_spec_loss + emotion_loss

                total_loss.backward()
                self.optimizer.step()





