import sys
from os.path import dirname, join, abspath
from model.multimodal import Multimodal
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
import parser_helper as helper
import time
import datetime
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from pathlib import Path

def get_checkpoint_path(checkpoint_path_str):
    """ Return the checkpoint path if it exists """
    if checkpoint_path_str == None:
        return None
    checkpoint_path = Path(checkpoint_path_str)

    if not checkpoint_path.exists():
        return None
    else:
        return checkpoint_path_str

class Solver(object):
    def __init__(self, config, train_loader, test_loader, train_eval, train_batch1):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_eval = train_eval
        self.train_batch1 = train_batch1

        self.config = config
        self.device = config.device #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.pretrained_check = get_checkpoint_path(config.pretrained_model)

        self.build_model()

        helper.logger('training', "[INFO] Building model")

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
        spec, spk_emb, phones, txt_feat, attn_mask, emotion_lb = vars

        #if state == "train":


        #elif state == "test":

        spec = spec.to(self.device)
        spk_emb = spk_emb.to(self.device)
        phones = phones.to(self.device)
        txt_feat = txt_feat.to(self.device)
        attn_mask = attn_mask.to(self.device)

        if self.device.type != "cpu": # 이거 용도 파악 정확히 못함
            spec = spec.type(torch.cuda.FloatTensor)
            spk_emb = spk_emb.type(torch.cuda.FloatTensor)
            phones = phones.type(torch.cuda.FloatTensor)
            txt_feat = txt_feat.type(torch.cuda.FloatTensor)
            attn_mask = attn_mask.type(torch.cuda.IntTensor)

        if state == "train":
            emotion_lb = emotion_lb.to(self.device)
            # emotion_lb = emotion_lb.type(torch.cuda.LongTensor)
            # 왜 IntTensor로 안하는것이지
        elif state == "test":
            pass

        return spec, spk_emb, phones, txt_feat, attn_mask, emotion_lb

    def calc_UA(self, logit, ground_truth):
        max_vals, max_indices = torch.max(logit, 1)
        ua = (max_indices == ground_truth).sum().numpy()/max_indices.size()[0]

        return ua

    def calc_WA(self, logit, ground_truth):
        n_tp = [0 for i in range(4)]
        n_tp_fn = [0 for k in range(4)]

        max_vals, max_indices = torch.max(logit, 1)
        tmp_correct = (max_indices.numpy() == ground_truth.numpy())
        for i, idx in enumerate(max_indices.numpy()):
            n_tp[idx] += tmp_correct[i]
            n_tp_fn[ground_truth.numpy()[i]] += 1

        return n_tp, n_tp_fn


    def uttr_eval(self, loader_type, epoch):
        uttr_eval_tp = [0 for i in range(self.config.n_classes)]
        uttr_eval_tp_fn = [0 for i in range(self.config.n_classes)]

        uttr_eval_ua = 0.0
        uttr_eval_wa = 0.0

        n_fold = self.config.test_dir.rsplit('/', 2)[1][4]

        with torch.no_grad():
            if loader_type == "train":
                data_loader = self.train_eval
            elif loader_type == "test":
                data_loader = self.test_loader

            helper.logger("info", "[INFO] Fold{} start segment-base evaluation...".format(n_fold))
            inf_start_time = time.time()

            self.model.eval()
            for batch_id, batch in enumerate(data_loader):
                self.optimizer.zero_grad()

                if self.config.speech_input == "wav2vec":
                    spec, spk_emb, phones, txt_feat, attn_mask, emotion_lb, wav2vec_feat = batch.values()
                elif self.config.speech_input == "spec":
                    spec, spk_emb, phones, txt_feat, emotion_lb = batch.values()
                    wav2vec_feat = None

                spk_emb = spk_emb.to(self.device)
                txt_feat = txt_feat.to(self.device)
                attn_mask = attn_mask.to(self.device)
                if self.device.type != "cpu":
                    spk_emb = spk_emb.type(torch.cuda.FloatTensor)
                    txt_feat = txt_feat.type(torch.cuda.FloatTensor)
                    attn_mask = attn_mask.type(torch.cuda.IntTensor)

                emo_preds = list()

                for spec, wav2vec_feat in zip(spec, wav2vec_feat):
                    spec = spec.to(self.device)
                    wav2vec_feat = wav2vec_feat.to(self.device)
                    if self.device.type != "cpu":
                        spec = spec.type(torch.cuda.FloatTensor)
                        wav2vec_feat = wav2vec_feat.to(self.device)

                    emo_pred = self.model(spec, spk_emb, None, None, wav2vec_feat, txt_feat, attn_mask)

                    emo_pred = emo_pred.detach().cpu()[0]
                    if len(emo_preds) == 0:
                        emo_preds = emo_pred
                    else:
                        emo_preds = [x+y for x,y in zip(emo_pred, emo_preds)]

                if(len(emo_preds) > 0):
                    emo_preds = torch.unsqueeze(torch.tensor(emo_preds),0)
                    uttr_eval_ua += self.calc_UA(emo_preds, emotion_lb)

                    tmp_tp, tmp_tp_fn = self.calc_WA(emo_preds, emotion_lb)
                    uttr_eval_tp = [x + y for x, y in zip(uttr_eval_tp, tmp_tp)]
                    uttr_eval_tp_fn = [x + y for x, y in zip(uttr_eval_tp_fn, tmp_tp_fn)]

            # 임시적으로 사용 코드 테스트할 때 분모가 0임을 방지
            if 0 in uttr_eval_tp_fn:
                for i, value in enumerate(uttr_eval_tp_fn):
                    if value == 0:
                        uttr_eval_tp_fn[i] = 1


            uttr_eval_ua = round(uttr_eval_ua / (batch_id + 1), 4)
            uttr_eval_wa = round(sum([x / y for x, y in zip(uttr_eval_tp, uttr_eval_tp_fn)]) / len(uttr_eval_tp), 4)

            print("[fold{} {} eval epoch{} UA {} eval WA {}]".format(n_fold, loader_type, epoch+1, uttr_eval_ua, uttr_eval_wa))
            inf = time.time() - inf_start_time
            inf = str(datetime.timedelta(seconds=inf))[:-7]
            helper.logger("info", "[TIME] Eval inference time {}".format(inf))
            helper.logger("info", "[EPOCH] [fold{} {} eval epoch{} UA {} WA {}]".format(n_fold, loader_type, epoch+1, uttr_eval_ua, uttr_eval_wa))

        return uttr_eval_ua, uttr_eval_wa
    '''
    def eval(self, loader_type):

        eval_tp = [0 for i in range(self.config.n_classes)]
        eval_tp_fn = [0 for i in range(self.config.n_classes)]

        eval_ua = 0.0

        n_fold = self.config.test_dir.rsplit('/', 2)[1]

        with torch.no_grad():
            if loader_type == "train":
                data_loader = self.train_batch1
            elif loader_type == "test":
                data_loader = self.test_loader

            helper.logger("info", "[INFO] Fold{} start segment-base evaluation...".format(n_fold))
            inf_start_time = time.time()

            self.model.eval()

            for batch_id, batch in enumerate(data_loader):
                self.optimizer.zero_grad() # 여기서 하는것 크게 의미 없긴함

                if self.config.speech_input == "wav2vec":
                    spec, spk_emb, phones, txt_feat, attn_mask, emotion_lb, wav2vec_feat = batch.values()
                    wav2vec_feat = wav2vec_feat.to(self.device)
                    if self.device.type != "cpu":
                        wav2vec_feat = wav2vec_feat.type(torch.cuda.FloatTensor)
                elif self.config.speech_input == "spec":
                    spec, spk_emb, phones, txt_feat, emotion_lb = batch.values()
                    wav2vec_feat = None

                tmp_vars = self.data_to_device(vars=[spec, spk_emb, phones, txt_feat, emotion_lb],
                                               state="test")
                spec, spk_emb, phones, txt_feat, emotion_lb = tmp_vars

                emotion_logits = self.model(spec, spk_emb, None, phones, wav2vec_feat, txt_feat, attn_mask)

                emotion_logits = emotion_logits.detach().cpu()
                # emotion_lb = emotion_lb.detach().cpu()

                eval_ua += self.calc_UA(eval_ua)

                tmp_tp, tmp_tp_fn = self.calc_WA(emotion_logits, emotion_lb)
                eval_tp = [x + y for x, y in zip(eval_tp, tmp_tp)]
                eval_tp_fn = [x + y for x, y in zip(eval_tp_fn, tmp_tp_fn)]

                # 일단 .detach .data() .cpu() .numpy() 이런것들을 하는데 마지막에 결과 나오고 하던데 용도 알아보기

            # eval_wa = sum([x/y for x,y in zip(train_tp, train_tp_fn)]) / len(train_tp)  # average recall per class
            print("[fold{} eval UA {} eval WA {}]".format(n_fold, eval_ua / (batch_id + 1),sum([x / y for x, y in zip(eval_tp, eval_tp_fn)]) / len(eval_tp)))
            inf = time.time() - inf_start_time
            inf = str(datetime.timedelta(seconds=inf))[:-7]
            helper.logger("info", "[TIME] Eval inference time {}".format(inf))
            helper.logger("info", "[EPOCH] [fold{} eval UA {} WA {}]".format(n_fold, eval_ua / (batch_id + 1), sum([x / y for x, y in zip(eval_tp, eval_tp_fn)]) / len(eval_tp)))
    '''
    def train(self):
        data_loader = self.train_loader
        if not Path(self.config.rs_save_path).exists():
            eval_records = pd.DataFrame(index=['fold', 'epoch', 'data_type','batch','UA','WA'])
        else:
            eval_records = pd.read_csv(self.config.rs_save_path)

        n_fold = self.config.train_dir.rsplit('/', 2)[1][4]
        helper.logger("info","[INFO] Fold{} start training...".format(n_fold))
        start_time = time.time()

        for epoch in range(self.config.epochs):
            epc_start_time = time.time()

            # To calculate Weighted Accuracy
            train_tp = [0 for i in range(self.config.n_classes)]
            train_tp_fn = [0 for i in range(self.config.n_classes)]
            test_tp = [0 for i in range(self.config.n_classes)]
            test_tp_fn = [0 for i in range(self.config.n_classes)]

            # To calculate Unweighted Accuracy
            train_ua = 0.0
            train_wa = 0.0
            test_ua = 0.0
            test_wa = 0.0


            self.model.train()

            for batch_id, batch in enumerate(tqdm(data_loader)):
                self.optimizer.zero_grad()

                if self.config.speech_input == "wav2vec":
                    spec, spk_emb, phones, txt_feat, attn_mask, emotion_lb, wav2vec_feat = batch.values()
                    wav2vec_feat = wav2vec_feat.to(self.device)
                    if self.device.type != "cpu":
                        wav2vec_feat = wav2vec_feat.type(torch.cuda.FloatTensor)
                elif self.config.speech_input == "spec":
                    spec, spk_emb, phones, txt_feat, emotion_lb = batch.values()
                    wav2vec_feat = None

                tmp_vars = self.data_to_device(vars=[spec, spk_emb, phones, txt_feat, attn_mask, emotion_lb],
                                               state="train")
                spec, spk_emb, phones, txt_feat, attn_mask, emotion_lb = tmp_vars

                spec_out, post_spec_out, emotion_logits = self.model(spec, spk_emb, spk_emb, phones, wav2vec_feat, txt_feat, attn_mask)

                spec_loss = F.mse_loss(spec, spec_out)
                post_spec_loss = F.mse_loss(spec, post_spec_out)

                emotion_loss = self.loss_fn(emotion_logits, emotion_lb)

                total_loss = spec_loss + post_spec_loss + emotion_loss

                total_loss.backward()
                self.optimizer.step()

                emotion_logits = emotion_logits.detach().cpu()
                emotion_lb = emotion_lb.detach().cpu()

                train_ua += self.calc_UA(emotion_logits, emotion_lb)

                tmp_tp, tmp_tp_fn = self.calc_WA(emotion_logits, emotion_lb)
                train_tp = [x+y for x,y in zip(train_tp, tmp_tp)]
                train_tp_fn = [x+y for x,y in zip(train_tp_fn, tmp_tp_fn)]


                if (batch_id+1) % self.config.log_interval == 0:
                    print("epoch {} batch id {} cls_loss {} const_loss {} train UA {}".format(epoch + 1, batch_id + 1,
                                                                                              emotion_loss.data.cpu().numpy(),
                                                                                              (spec_loss + post_spec_loss)/2,
                                                                                              train_ua / (batch_id + 1)))
                    helper.logger("training", "[INFO] epoch {} batch id {} cls_loss {} spec_loss {} post_spec_loss {} const_loss {} train UA {}".format(epoch + 1, batch_id + 1,
                                                                                                                         emotion_loss.data.cpu().numpy(),
                                                                                                                         spec_loss, post_spec_loss,
                                                                                                                         (spec_loss + post_spec_loss)/2,
                                                                                                                         train_ua / (batch_id + 1)))

            # 임시적으로 사용 코드 테스트할 때 분모가 0임을 방지
            if 0 in train_tp_fn:
                for i, value in enumerate(train_tp_fn):
                    if value == 0:
                        train_tp_fn[i] = 1

            train_ua = round(train_ua / (batch_id + 1),4)
            train_wa = round(sum([x/y for x,y in zip(train_tp, train_tp_fn)]) / len(train_tp), 4)  # average recall per class # 소수점 4자리까지 출력
            print("[fold{} epoch {} train UA {} WA {}]".format(n_fold, epoch + 1, train_ua, train_wa))
            epc = time.time() - epc_start_time
            epc = str(datetime.timedelta(seconds=epc))[:-7]
            helper.logger("info", "[TIME] epoch {} training time {}".format(epoch + 1 , epc))
            helper.logger("info","[EPOCH] [fold{} epoch {} train UA {} train WA {}]".format(n_fold, epoch + 1, train_ua, train_wa))


            train_eval_ua, train_eval_wa = self.uttr_eval(loader_type = "train", epoch=epoch)
            test_ua , test_wa = self.uttr_eval(loader_type = "test", epoch = epoch)

            # ['fold', 'epoch', 'data_type','batch','UA','WA']
            insert_data = {'fold':n_fold,'epoch':epoch+1, 'data_type':"test",'batch':self.config.batch_size, 'UA':test_ua, 'WA':test_wa}
            eval_records.append(insert_data, ignore_index=True)

            torch.save({"epoch": (epoch + 1),
                        "model": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()},
                       str(self.config.md_save_dir) + "/checkpoint_step_" + str(epoch + 1) + "_neckdim_" + str(
                           self.config.dim_neck) + ".ckpt")
        eval_records.to_csv(self.config.rs_save_path, index=False)
