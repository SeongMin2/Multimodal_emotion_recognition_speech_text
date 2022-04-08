# import sys
# from os.path import dirname, join, abspath
# sys.path.insert(0, abspath(join(dirname(__file__), '')))
from audio.data_loader import get_data_loaders, get_train_data_loaders
import parser_helper as helper
from model.multimodal import Multimodal

from torch.backends import cudnn

# Name of the train and test pkl files
train_npz = "train"
test_npz = "test"



def model_check(model, train_loader):
    data_iter = iter(train_loader)
    feat = next(data_iter)
    # {
    #                 "spec": features["spec"],   # (batch,96,80)
    #                 "spk_emb": features["spk_emb"], # (batch, 256)
    #                 "phones": features["phones"],   # (batch, 96, 128)
    #                 "emotion_lb": features["emotion_lb"], # (batch)
    #                 "text": features["text"],          # 이 친구는 list len은 batch 수
    #                 "wav2vec_feat": features["wav2vec_feat"], # (batch, 25, 96, 1024)
    #                 "txt_feat" : features["txt_feat"]  # (batch, 124, 768(or 1024) )
    #             }
    result = model(feat['spec'], feat['spk_emb'], feat['spk_emb'], feat['phones'], feat['wav2vec_feat'], feat['txt_feat'])
    # def forward(self, spec, spk_emb, phones, wav2vec_feat, txt_feat):





def main():
    config = helper.get_training_config()
    train_loader = get_train_data_loaders(config, train_npz)
    # train_loader, test_loader, train_eval, train_1batch = get_data_loaders(config, train_npz, test_npz)

    model = Multimodal(config)
    model_check(model, train_loader)
    pass




if __name__ == '__main__':
    main()