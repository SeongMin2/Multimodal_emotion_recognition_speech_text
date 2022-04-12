# import sys
# from os.path import dirname, join, abspath
# sys.path.insert(0, abspath(join(dirname(__file__), '')))
from audio.data_loader import get_data_loaders, get_train_data_loaders
import parser_helper as helper
from model.multimodal import Multimodal
from solver import Solver

from torch.backends import cudnn

# Name of the train and test pkl files
train_npz = "train"
test_npz = "test"

def get_solver(config, train_loader, test_loader, train_eval, train_batch1):
    solver = Solver(config, train_loader, test_loader, train_eval, train_batch1)

    helper.logger("info", "[INFO] Getting the model...")

    return solver

def model_check(model, train_loader):
    data_iter = iter(train_loader)
    feat = next(data_iter)
    # return {
    #                 "spec": features["spec"],
    #                 "spk_emb": features["spk_emb"],
    #                 "phones": features["phones"],
    #                 "text": features["text"],
    #                 "txt_feat" : features["txt_feat"],
    #                 "emotion_lb": features["emotion_lb"],
    #                 "wav2vec_feat": features["wav2vec_feat"]
    #             }
    result = model(feat['spec'], feat['spk_emb'], feat['spk_emb'], feat['phones'], feat['wav2vec_feat'], feat['txt_feat'])
    # def forward(self, spec, spk_emb, phones, wav2vec_feat, txt_feat):





def main():
    config = helper.get_training_config()
    train_loader, test_loader = get_train_data_loaders(config, train_npz, test_npz)
    #train_loader, test_loader, train_eval, train_batch1 = get_data_loaders(config, train_npz, test_npz)
    helper.logger("info", "[INFO] Data loading complete!")
    solver = get_solver(config, train_loader, test_loader, None, None)#test_loader, train_eval, train_batch1)
    #solver.uttr_eval("test")
    solver.train()
    model = Multimodal(config)
    model_check(model, train_loader)
    pass


if __name__ == '__main__':
    main()