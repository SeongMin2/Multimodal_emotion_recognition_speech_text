import torch
import torch.nn as nn
import model.models as models

def get_SER_Tail(config):
    return models.SERTail(config)

def get_Encoder(config):
    return models.Encoder(config)

def get_Decoder(config):
    return models.Decoder(config)

def get_PostNet(config):
    return models.Postnet(config)

def get_Phone_Encoder(config):
    return models.PhoneEncoder(config)

def get_Classifier(config):
    return models.SingleClassifier(config)

class SER(nn.Module):
    def __init__(self, config):
        super(SER, self).__init__()

        self.device = torch.device("cuda:" + str(config.n_cuda_device) if torch.cuda.is_available() else 'cpu')

        self.encoder = get_Encoder(config)
        self.ser_tail = get_SER_Tail(config)
        self.phone_encoder = get_Phone_Encoder(config)

        self.decoder = get_Decoder(config)
        self.post_net = get_PostNet(config)

        self.glob_max_pool_s = nn.MaxPool1d(kernel_size=config.len_crop, stride=config.len_crop)

        self.classifier = get_Classifier(config)

        self.speech_input = config.speech_input
        self.dim_neck = config.dim_neck
        self.freq = config.freq

    def forward(self, spec, spk_emb, spk_emb_dc, phones, wav2vec_feat):
        encoder_outputs = self.encoder(spec, spk_emb, wav2vec_feat)
        # encoder_outputs : (batch, 96, 2*d)
        ser_feat = self.ser_tail(encoder_outputs)

        src_s = self.glob_max_pool_s(ser_feat.transpose(1, 2))
        src_s = src_s.view(src_s.shape[0], -1)

        output = self.classifier(src_s)

        # for inference
        if spk_emb_dc is None:
            # return codes and logits
            return output

        # AutoEncoder for disentanglement learning
        ##########################################
        #              Down-Sampling             #
        ##########################################

        # downsampling
        codes = []
        # encoder_outputs = encoder_outputs * self.delta
        out_forward = encoder_outputs[:, :, :self.dim_neck]
        out_backward = encoder_outputs[:, :, self.dim_neck:]

        # downsample
        for i in range(0, encoder_outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))

        ##########################################
        #              Up-Sampling               #
        ##########################################

        tmp = []

        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(spec.size(1) / len(codes)), -1))
        up_sampled = torch.cat(tmp, dim=1)
        # up_sampled : (batch, 96, 16)

        phone_feat = self.phone_encoder(phones)
        # phone_feat : (batch, 96, 256)

        decoder_input = torch.cat((up_sampled, spk_emb_dc.unsqueeze(1).expand(-1, spec.size(1), -1), phone_feat),
                                  dim=-1)
        # decoder_input : (batch, 96, 512 + 2*d)

        decoder_output = self.decoder(decoder_input)
        # spec_logits : (batch, 96, 80)
        # spec : (batch, 96, 80)

        post_output = self.post_net(decoder_output.transpose(1, 2))
        # post_output : (batch, 80, 96)

        return decoder_output, post_output.transpose(1, 2), output