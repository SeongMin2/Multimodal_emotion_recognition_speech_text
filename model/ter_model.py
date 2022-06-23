import torch
import torch.nn as nn
import model.models as models

def get_TER(config):
    return models.TxtModel(config)

def get_Classifier(config):
    return models.SingleClassifier(config)

class TER(nn.Module):
    def __init__(self, config):
        super(TER, self).__init__()

        self.device = torch.device("cuda:" + str(config.n_cuda_device) if torch.cuda.is_available() else 'cpu')

        self.txt_model = get_TER(config)

        self.classifier = get_Classifier(config)

        self.glob_max_pool_t = nn.MaxPool1d(kernel_size=config.max_token_len - 2, stride=config.max_token_len - 2)

    def forward(self, txt_feat):
        ter_feat = self.txt_model(txt_feat)

        src_t = self.glob_max_pool_t(ter_feat.transpose(1, 2))
        src_t = src_t.view(src_t.shape[0], -1)

        output = self.classifier(src_t)

        return output