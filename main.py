import torch
from audio.data_loader import get_data_loaders, get_train_data_loaders
import parser_helper as helper
from solver import Solver
import random
import numpy as np
import warnings
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

def set_seed(seed: int):
    torch.manual_seed(seed)  # 이놈이 초기 weight 값 들도 모두 고정 시킴
    '''
    이거 두개는 연산 속도 느려져서 연구 실험 후반 단계에 사용하라고 권장함
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    음 CuDNN의 randomness를 제어함
    '''
    random.seed(seed)
    np.random.seed(seed)

    # 동일한 조건에서 학습 시 weight가 변화하지 않게 하는 옵션
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU

def main():
    warnings.filterwarnings(action='ignore') # 일단 pretrained 불러서 쓸 때마다 userwarning 나오고 padding="same"관련해서 나오는데 무시하도록..
    # 근데 pretrained 관련된 warning은 무시 안되네
    
    config = helper.get_training_config()
    set_seed(config.seed)
    #train_loader, test_loader = get_train_data_loaders(config, train_npz, test_npz)
    train_loader, test_loader, train_eval = get_data_loaders(config, train_npz, test_npz)
    helper.logger("info", "[INFO] Data loading complete!")
    solver = get_solver(config, train_loader, test_loader, train_eval, None) #test_loader, train_eval, train_batch1)
    #solver.uttr_eval("test",1)
    solver.train()

if __name__ == '__main__':
    main()