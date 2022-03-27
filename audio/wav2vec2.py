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