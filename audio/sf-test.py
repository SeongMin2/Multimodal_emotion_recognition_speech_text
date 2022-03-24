import soundfile as sf
import librosa
sr = 16000
data, samplerate = sf.read('../IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav')
# default가 16000인듯 왜 근데 공홈에 뚜렷히 안나와 있지..

vector, _sr = librosa.load('../IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav',sr=sr)

pass