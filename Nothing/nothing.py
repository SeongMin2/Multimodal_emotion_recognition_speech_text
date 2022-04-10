import os
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.functional as F
#import torchtext
#from torchtext import data, datasets
import random
import numpy as np
#from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoConfig

a=[1,2,3]
b=[4,5,6]
c = [a+b for a,b in zip(a,b)]

def calc_WA(logit, ground_truth):
    n_tp = [0 for i in range(4)]
    n_tp_fn = [0 for k in range(4)]

    max_vals, max_indices = torch.max(logit, 1)
    tmp_correct = (max_indices.numpy() == ground_truth.numpy())
    for i, idx in enumerate(max_indices.numpy()):
        n_tp[idx] += tmp_correct[idx]
        n_tp_fn[ground_truth[i]] += 1

    return n_tp, n_tp_fn



def calc_UA(logit, ground_truth):
    max_vals, max_indices = torch.max(logit, 1)
    tmp = (max_indices == ground_truth)
    ua = (max_indices == ground_truth).sum().data.cpu().numpy() / max_indices.size()[0]

    return ua
logit = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
ground_truth = torch.tensor([0,3,2])
n_tp, n_tp_fn = calc_WA(logit, ground_truth)
ua = calc_UA(logit, ground_truth)

# text_model = AutoModel.from_pretrained("bert-base-uncased")
# config = AutoConfig.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
encoded_input = tokenizer("hello who are you", padding="max_length", max_length=10, return_tensors='pt')



a = torch.rand(3,4,5)
b = a.sum(dim=1)
k = torch.cat((a,a), dim=2)
pass
'''
a = torch.rand(3,5)
c = a.sum(dim=1)

b = nn.Linear(5,1)
r = b(a)
r2 = F.softmax(r, dim=0)
r3 = a* r2

alpha = torch.rand(2,3)

alpha = torch.tensor(alpha)
sm_reult = F.softmax(alpha,dim=1)


input = torch.zeros(2,4)

input = input + 2

result = alpha * input

layer1 = nn.Linear(5,6)
output = layer1(input)
'''
'''
a = [[1,2],[2,3]]
b = [[4,3], [4,5]]
c = a+b
pass
'''

'''
a = 1
if a == 1 or 2:
    print('ya')

print('I can do it!')
'''

'''
a = np.load("./full_data/train.npz", allow_pickle=True)
print(a)
pass
'''

'''
# 절대 경로 실험
ABS_PATH = str(Path(".").absolute()) # 이 absolute는 이 코드가 위치하는 파일 기준이 아니라 실행되는 코드 기준의 경로임
abs = 'C:\\SPB_Data\\iemocap_preprocessing'
ab = 'C:\SPB_Data\iemocap_preprocessing'
print(abs)
pass
'''

'''
a = [['hello','a'],['bye','b']]
np.savez('test.npz',a)

a_load = np.load('test.npz')

print(a_load.files)

a_load.close()
'''

'''
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10

TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(trainset, min_freq=5) # 단어 집합 생성
LABEL.build_vocab(trainset)


vocab_size = len(TEXT.vocab)
n_classes = 2
print('단어 집합의 크기 : {}'.format(vocab_size))
print('클래스의 개수 : {}'.format(n_classes))


trainset, valset = trainset.split(split_ratio=0.8)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, testset), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)

class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        x, _ = self.gru(x, h_0)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        h_t = x[:,-1,:] # (배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        self.dropout(h_t)
        logit = self.out(h_t)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

model = GRU(1, 256, vocab_size, 128, n_classes, 0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text, batch.label
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
'''

# wav2vec text
'''
# https://huggingface.co/facebook/wav2vec2-large-lv60
from transformers import AutoProcessor, AutoModelForPreTraining # 여기서 왜 AutoProcessor 머가 문제야
import librosa
import torch

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-lv60")

model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-lv60")
print(model)
# 해당 모델이 16K sampling rate로 학습시켰기 때문

sr = 16000
vector, _sr = librosa.load('../IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav',sr=sr)

vector = torch.tensor(vector)
vector = torch.unsqueeze(vector,0)
result = model(vector)

pass
'''

# soundfild test
'''
import soundfile as sf
import librosa
sr = 16000
data, samplerate = sf.read('../IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav')
# default가 16000인듯 왜 근데 공홈에 뚜렷히 안나와 있지..

vector, _sr = librosa.load('../IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav',sr=sr)

pass
'''