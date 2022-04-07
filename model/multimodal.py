import torch
import torch.nn as nn
import sys
import model.models as models

def get_SER_Tail(config):

    input_len = 2 * config.dim_neck

    use_sigmoid = False

    ser_tail_model = models.SER_Tail(input_len=input_len,
                               use_drop=True,
                               use_sigmoid = use_sigmoid)

    return ser_tail_model

def get_Encoder(config):
    return models.Encoder(config)

def get_Decoder(config):
    return models.Decoder(config)

def get_Phone_Encoder(config):
    return models.PhoneEncoder(config)

def get_TER(config):
    return models.TxtModel(config)

class Multimodal(nn.Module):
    def __init__(self, config):
        super(Multimodal).__init__()

        self.encoder = get_Encoder(config)

        self.phone_encoder = get_Phone_Encoder(config)

        self.decoder = get_Decoder(config)

        self.ser_tail = get_SER_Tail(config)

        self.txt_model = get_TER(config)

        self.speech_input = config.speech_input

    def forward(self, spec, spk_emb, phones, wav2vec_feat, txt_feat):
        encoder_output = self.encoder(spec, spk_emb, wav2vec_feat)
        
        phone_feat = self.phone_encoder(phones)
        
        # decoder input 넣기 전에 upsampling이랑 concatenation 하고 넣어야함

        ser_feat = self.ser_tail(encoder_output)
        
        ter_feat = self.txt_model(txt_feat)
        
        







