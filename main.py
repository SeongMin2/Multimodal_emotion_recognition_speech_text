# import sys
# from os.path import dirname, join, abspath
# sys.path.insert(0, abspath(join(dirname(__file__), '')))
import torch
from audio.data_loader import get_data_loaders, get_train_data_loaders
import parser_helper as helper
from model.multimodal import Multimodal
from solver import Solver
import random
import numpy as np

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

def set_seed(seed: int = 42):
    torch.manual_seed(1)  # 이놈이 초기 weight 값 들도 모두 고정 시킴
    '''
    이거 두개는 연산 속도 느려져서 연구 실험 후반 단계에 사용하라고 권장함
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    음 CuDNN의 randomness를 제어함
    '''
    random.seed(1)
    np.random.seed(1)

    # 동일한 조건에서 학습 시 weight가 변화하지 않게 하는 옵션
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)  # if use multi-GPU

def main():
    set_seed(42)

    config = helper.get_training_config()
    #train_loader, test_loader = get_train_data_loaders(config, train_npz, test_npz)
    train_loader, test_loader, train_eval= get_data_loaders(config, train_npz, test_npz)
    helper.logger("info", "[INFO] Data loading complete!")
    solver = get_solver(config, train_loader, test_loader, train_eval, None)#test_loader, train_eval, train_batch1)
    solver.train()
    #model = Multimodal(config)
    #model_check(model, train_loader)
    pass


if __name__ == '__main__':
    main()
    # 여기다 seed를 처리하면 안됨, 모델에 닿지도 않음