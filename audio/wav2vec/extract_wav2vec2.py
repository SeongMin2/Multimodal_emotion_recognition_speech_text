from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import sys, os
import numpy as np
from spectrogram_helpers import get_spec
# 이거 오류 없이 잘 됨 왜 줄 뜨는지 몰겠음

MODEL_TYPE = "facebook/wav2vec2-large-lv60"

processor = Wav2Vec2Processor.from_pretrained(MODEL_TYPE) # normalize the data
model = Wav2Vec2ForCTC.from_pretrained(MODEL_TYPE)

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
    speech = np.pad(batch, padding, mode='constant')  # 앞 뒤로 160씩 총 320 padding
    inputs = processor(speech, return_tensors="pt", sampling_rate=sr, padding=True)
    # 일단 여기서 저 padding=True가 뭔지 모르겠음
    # https://huggingface.co/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad.padding
    # 그냥 혼자 padding할거면 의미 없음
    # 아니 이 논문 코드 wav2vec2 이해 안되네 왜 padding하고 normalization하는거지 이게 맞는건가 그리고 왜 attention mask는 padding 부분은 0으로 안잡고 하는거지
    # 걍 없애도 되는듯

    input_values = inputs.input_values
    attention_mask = inputs.attention_mask

    # we run the model in inference mode just to get the output
    with torch.no_grad():
        output = model(input_values, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = get_hidden_states(batch, output.hidden_states)

    return hidden_states

# 실험
'''
sr = 16000
speech, _sr = librosa.load('../../IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav', sr=sr)
feature = apply_wav2vec(speech,sr)
# feature의 shape는 (25,98,1024) 잘 나옴
print('hi')
pass
'''



