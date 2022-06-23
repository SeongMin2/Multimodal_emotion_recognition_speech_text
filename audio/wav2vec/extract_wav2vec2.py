from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
#from torch.nn.parallel import data_parallel
import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import parser_helper as helper
from .spectrogram_helpers import get_spec

config = helper.get_training_config()
MODEL_TYPE = config.pretrained_wav2vec2_model

processor = Wav2Vec2Processor.from_pretrained(MODEL_TYPE) # normalize the data
model = Wav2Vec2ForCTC.from_pretrained(MODEL_TYPE)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:"+str(config.n_cuda_device) if torch.cuda.is_available() else 'cpu')
#device = config.device
print("Device: ", device)

model = model.to(device)

def get_hidden_states(batch, hidden_states):
    spec = get_spec(batch)
    
    if not (spec.shape[0] == hidden_states[-1].detach().cpu().numpy().shape[1]):
        #print("speech file: ", speech_file)
        #print("speech len: ", len(sp))
        # 음 이건 나중에 처리
        print("spec: ", spec.shape)
        print("features: ", hidden_states[-1].detach().cpu().numpy().shape)
        sys.exit(1)

        # make a list of numpy arrays
    tmp_states = []
    for element in hidden_states:
        # transform the tensors to numpy and append to the list
        tmp_states.append(element.detach().cpu().numpy())

    # transform the list of numpy into numpy
    tmp_states = np.concatenate(tmp_states, axis=0)  # shape는 (25,98,1024)

    return tmp_states


def apply_wav2vec(batch, sr:int):
    # 일단 이 padding도 의도는 모르겠으나 논문 한 사람들이 그렇다 하니 의미는 나중에 보기

    if (len(batch) % 320 >= 80):
        padding = 160
    else:
        padding = 319
        # pad the speech data
    # for zip으로 batch안에 있는 놈들 적용하려고 했는데 그럴필요 없을 지도

    speech = np.pad(batch, padding, mode='constant')  # 앞 뒤로 160씩 총 320 padding
    padding_mask = torch.tensor([1 for i in range(len(speech))])
    padding_mask[0:padding] = 0
    padding_mask[(len(speech)-padding):len(speech)] = 0
    padding_mask = padding_mask.unsqueeze(0)

    inputs = processor(speech, return_tensors="pt", sampling_rate=sr, padding=True) # userwarning 나오는데 이거 내부 문제임
    # 일단 여기서 저 padding=True가 뭔지 모르겠음
    # https://huggingface.co/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad.padding
    # 그냥 혼자 padding할거면 의미 없음
    # 아니 이 논문 코드 wav2vec2 이해 안되네 왜 padding하고 normalization하는거지 이게 맞는건가 그리고 왜 attention mask는 padding 부분은 0으로 안잡고 하는거지
    # 걍 없애도 되는듯

    input_values = inputs.input_values
    attention_mask = inputs.attention_mask

    attention_mask = attention_mask * padding_mask

    input_values = input_values.to(device)
    attention_mask = attention_mask.to(device)


    # we run the model in inference mode just to get the output
    with torch.no_grad():
        output = model(input_values, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = get_hidden_states(batch, output.hidden_states)

    return hidden_states  # 이거 detach()의미 갔다와서 더 보고 cpu에 빼는게 나을지 생각해보기

# 실험
'''
sr = 16000
speech, _sr = librosa.load('../../IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav', sr=sr)
feature = apply_wav2vec(speech,sr)
# feature의 shape는 (25,98,1024) 잘 나옴 # 아 98부분은 파일에 따라서 다름
print('hi')
pass
'''