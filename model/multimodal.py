import torch
import torch.nn as nn
import sys
import model.models as models
import model.attention as attention

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

def get_PostNet(config):
    return models.Postnet(config)

def get_Phone_Encoder(config):
    return models.PhoneEncoder(config)

def get_TER(config):
    return models.TxtModel(config)

def get_Cross_attention(config):
    return attention.Cross_attention(config)

class Multimodal(nn.Module):
    def __init__(self, config):
        super(Multimodal).__init__()

        self.encoder = get_Encoder(config)
        self.ser_tail = get_SER_Tail(config)
        self.phone_encoder = get_Phone_Encoder(config)

        self.decoder = get_Decoder(config)
        self.post_net = get_PostNet(config)

        self.txt_model = get_TER(config)

        self.cross_attention = get_Cross_attention(config)

        self.speech_input = config.speech_input

    def forward(self, spec, spk_emb, phones, wav2vec_feat, txt_feat):
        encoder_outputs = self.encoder(spec, spk_emb, wav2vec_feat)

        ##########################################
        #              Down-Sampling             #
        ##########################################

        # downsampling
        codes = []
        out_forward = encoder_outputs[:, :, :self.dim_neck]
        out_backward = encoder_outputs[:, :, self.dim_neck:]

        # downsample
        for i in range(0, encoder_outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))

        down_sampled = codes

        ##########################################
        #              Up-Sampling               #
        ##########################################

        tmp = []

        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(spec.size(1)/len(codes)),-1))
        up_sampled = torch.cat(tmp, dim=1)

        phone_feat = self.phone_encoder(phones)

        decoder_input = torch.cat((up_sampled, spk_emb.unsqueeze(1).expand(-1,spec.size(1),-1), phone_feat), dim=-1)
        
        # decoder input 넣기 전에 upsampling이랑 concatenation 하고 넣어야함

        decoder_output = self.decoder(decoder_input)

        post_output = self.post_net(decoder_output)

        ser_feat = self.ser_tail(up_sampled)
        
        ter_feat = self.txt_model(txt_feat)

        cross_attn1 = self.cross_attention(ser_feat, ter_feat, ter_feat)

        cross_attn2 = self.cross_attention(ter_feat, ser_feat, ser_feat)

        # cross_attn 값들을 각각 nn.Linear로 한 번 더 거치고 나서 concatenation 하고 내보내자

        output = 1

        return 1


        
        







