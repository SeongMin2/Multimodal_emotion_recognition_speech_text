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

        if not (self.pretrained_check == None):  # 만약 pretrained 된 모델 파일이 존재한다면,
            #with open(self.test_out_file, "a") as txt_f:
            #    txt_f.write("Loading the weights at " + self.checkpoint_path + "\n")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def data_to_device(self, vars):
        spec, spk_emb, phones, emotion_lb = vars


    # def Build_model(self):

