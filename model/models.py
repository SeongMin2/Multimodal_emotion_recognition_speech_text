import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class WeightedAvg(torch.nn.Module):
    def __init__(self):
        super(WeightedAvg, self).__init__()

        weights = torch.ones([1, 25, 1, 1])
        self.weights = nn.Parameter(weights, requires_grad=True)

    def forward(self, signal):
        tmp_mul = torch.mul(signal, self.weights)
        avg_signal = torch.div(torch.sum(tmp_mul, dim=1), torch.sum(self.weights, dim=1))
        return avg_signal

class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.speech_input = config.speech_input
        self.dim_neck = config.dim_neck
        self.freq = config.freq

        conv_dim = config.conv_dim

        if config.speech_input == "wav2vec":
            input_len = config.wav2vec_feat_len
            conv_dim = 2048
        elif config.speech_input == "spec":
            input_len = config.num_mels

        self.input_len = input_len + config.spk_feat_len

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.input_len if i == 0 else conv_dim,
                         conv_dim,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(conv_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        # lstm layer
        self.lstm = nn.LSTM(conv_dim, self.dim_neck, 2, batch_first=True, bidirectional=True)

        # average sum of wav2vec as speech input
        if ((self.speech_input == "wav2vec") or (self.speech_input == "spec_and_wav2vec")):
            self.weighted_avg_in = WeightedAvg()

    def forward(self, x, spk_org, wav2vec_feat):

        # # prepare the input
        x = x.squeeze(1).transpose(2, 1)

        # determine the speech input
        if (self.speech_input == "spec"):
            speech = x
        elif (self.speech_input == "wav2vec"):
            speech = self.weighted_avg_in(wav2vec_feat).squeeze(1).transpose(2, 1)
        elif (self.speech_input == "spec_and_wav2vec"):
            tmp_wav2vec = self.weighted_avg_in(wav2vec_feat).squeeze(1).transpose(2, 1)
            speech = torch.cat((x, tmp_wav2vec), dim=1)
        else:
            print("Error! Undefined speech input type")
            sys.exit(1)

        spk_org = spk_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        speech = torch.cat((speech, spk_org), dim=1)

        # apply convolutions
        for conv in self.convolutions:
            speech = F.relu(conv(speech))

        speech = speech.transpose(1, 2)

        # apply blstm
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(speech)

        # downsampling
        codes = []
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        # downsample
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))

        return codes


class PhoneEncoder(nn.Module):
    """ PhoneEncoder module:
    """

    def __init__(self, config):
        super(PhoneEncoder, self).__init__()

        # input to the phone encoder are phones
        input_dim = config.dim_phone
        conv_dim = 512

        # convolution layers
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(input_dim if i == 0 else conv_dim,
                         conv_dim,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(conv_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        # lstm layer
        self.lstm = nn.LSTM(conv_dim, config.dim_phone_emb, 2, batch_first=True, bidirectional=True)

    def forward(self, x):

        x = x.squeeze(1).transpose(2, 1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    """ Decoder module:
    """

    def __init__(self, config):
        super(Decoder, self).__init__()

        # define the input size
        lstm_in = config.dim_neck * 2 + config.dim_emb + config.dim_phone_emb * 2

        # first lstm layer
        self.lstm1 = nn.LSTM(lstm_in, config.dim_pre, 1, batch_first=True)

        # convolutions
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(config.dim_pre,
                         config.dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(config.dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        # second lstm layer
        self.lstm2 = nn.LSTM(config.dim_pre, 1024, 2, batch_first=True)

        self.linear_projection = LinearNorm(1024, config.num_mels)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x)

        decoder_output = self.linear_projection(outputs)

        return decoder_output


