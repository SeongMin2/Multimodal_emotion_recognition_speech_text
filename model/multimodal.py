import torch
import torch.nn as nn
import model.models as models
import model.attention as attention

def get_SER_Tail(config):
    return models.SERTail(config)

def get_SER_Tail_origin(config):
    input_len = 2 * config.dim_neck
    use_sigmoid = False

    ser_tail_model = models.SER_Tail_origin(input_len=input_len,
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

def get_Classifier(config):
    return models.Classifier(config)

def get_Cross_attention(config):
    return attention.Cross_attention(config)

def duplicate_n_heads_times(n_heads, feat):
    output = []

    # Duplicate original features by n_heads times
    for i in range(n_heads):
        if i == 0:
            output = feat
        else:
            output = torch.cat((output, feat), dim=-1)
    return output


class Multimodal(nn.Module):
    def __init__(self, config):
        super(Multimodal, self).__init__()

        self.device = config.device

        self.encoder = get_Encoder(config)
        self.ser_tail = get_SER_Tail(config)
        self.phone_encoder = get_Phone_Encoder(config)

        self.decoder = get_Decoder(config)
        self.post_net = get_PostNet(config)

        self.txt_model = get_TER(config)

        self.classifier = get_Classifier(config)

        self.cross_attn_s_md = get_Cross_attention(config)
        self.cross_attn_t_md = get_Cross_attention(config)

        # Layernorm은 default로 trainable 함 그래서 모두 따로 정의해줌
        self.layer_norm_s = nn.LayerNorm(config.attention_emb)
        self.layer_norm_t = nn.LayerNorm(config.attention_emb)
        self.post_layer_norm_s = nn.LayerNorm(config.attention_emb)
        self.post_layer_norm_t = nn.LayerNorm(config.attention_emb)

        self.ff_s = nn.Linear(config.attention_emb, config.attention_emb)
        self.ff_t = nn.Linear(config.attention_emb, config.attention_emb)

        self.glob_avg_pool_s = nn.AvgPool1d(kernel_size = config.len_crop , stride = config.len_crop) # 이거 crop size말고 specp[0] size로 바꾸는것이 좋음
        self.glob_avg_pool_t = nn.AvgPool1d(kernel_size = config.max_token_len-2, stride=config.max_token_len-2) # [cls] [eos] dataset에서 생성자 정의할 때 뺐음

        self.speech_input = config.speech_input
        self.dim_neck = config.dim_neck
        self.freq = config.freq
        self.len_crop = config.len_crop
        self.n_heads = config.n_heads
        self.batch_size = config.batch_size

    def make_attn_mask(self, len, mask_start_idx):
        attn_mask = [1 for i in range(len)]
        if ((len - 1) - mask_start_idx) >= 0 :
            for idx in range(mask_start_idx, len):
                attn_mask[idx] = 0

        return attn_mask

    def get_attn_mask(self, txt_len, spch_len, attn_mask_ids):
        txt_attn_mask = list()
        spch_attn_mask = list()
        for ids in attn_mask_ids:
            txt_attn_mask.append(self.make_attn_mask(txt_len, ids[0]))
            spch_attn_mask.append(self.make_attn_mask(spch_len, ids[1]))

        txt_attn_mask = torch.tensor(txt_attn_mask).to(self.device)
        spch_attn_mask = torch.tensor(spch_attn_mask).to(self.device)
        if self.device.type != "cpu": # 이거 용도 파악 정확히 못함
            txt_attn_mask = txt_attn_mask.type(torch.cuda.IntTensor)
            spch_attn_mask = spch_attn_mask.type(torch.cuda.IntTensor)

        return txt_attn_mask, spch_attn_mask


    def forward(self, spec, spk_emb, spk_emb_dc ,phones, wav2vec_feat, txt_feat, attn_mask_ids):

        encoder_outputs = self.encoder(spec, spk_emb, wav2vec_feat)
        # encoder_outputs : (batch, 96, 2*d)

        ser_feat = self.ser_tail(encoder_outputs)
        
        ter_feat = self.txt_model(txt_feat)

        ser_feat = ser_feat.transpose(1,2)
        ter_feat = ter_feat.transpose(1,2)

        txt_attn_mask, spch_attn_mask = self.get_attn_mask(txt_feat.shape[1], spec.shape[1],attn_mask_ids)

        cross_attn_s = self.cross_attn_s_md(ser_feat, ter_feat, ter_feat, txt_attn_mask)  # attn_mask는 key에 대해서 진행되는것임
        cross_attn_t = self.cross_attn_t_md(ter_feat, ser_feat, ser_feat, spch_attn_mask)
        # cross_attn2 : (batch, 122, 128)

        src_s = self.layer_norm_s(ser_feat + cross_attn_s)
        src_t = self.layer_norm_t(ter_feat + cross_attn_t)
        # src2 : (batch, 122, 128)

        src_s = self.post_layer_norm_s(src_s + self.ff_s(src_s))
        src_t = self.post_layer_norm_t(src_t + self.ff_t(src_t))

        src_s = self.glob_avg_pool_s(src_s.transpose(1,2))
        src_t = self.glob_avg_pool_t(src_t.transpose(1,2))

        src = torch.cat((src_s.view(src_s.shape[0], -1), src_t.view(src_t.shape[0], -1)), dim=-1)
        # 이후로 classifier 들어가면 됨

        output = self.classifier(src)

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

        return decoder_output, post_output.transpose(1,2), output


        
        







