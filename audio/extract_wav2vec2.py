from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
MODEL_TYPE = "facebook/wav2vec2-large-lv60"

processor = Wav2Vec2Processor.from_pretrained(MODEL_TYPE) # normalize the data
model = Wav2Vec2ForCTC.from_pretrained(MODEL_TYPE)

sr = 16000
vector, _sr = librosa.load('../IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav',sr=sr)

# vector = torch.unsqueeze(vector,0)
inputs = processor(vector, return_tensors="pt", sampling_rate=16000, padding=True)
pass