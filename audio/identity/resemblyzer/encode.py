import numpy as np
from resemblyzer import preprocess_wav, VoiceEncoder
from tqdm import tqdm

def get_speaker_embeddings(dir_name, speaker, wav_list, idx_uttrs, num_uttrs):  # speaker embedding 추출
    spk_encoder = VoiceEncoder()

    # make list with num_uttrs random wave files
    wav_files = []
    for i in range(num_uttrs):
        file_name = wav_list[idx_uttrs[i]][:-4] + ".wav"
        wav_file = str(dir_name) + "/" + str(speaker) + "/" + str(file_name)
        wav_files.append(wav_file)

    speaker_wavs = np.array(list(map(preprocess_wav, tqdm(wav_files, "Preprocessing wavs", len(wav_files)))))
    utterance_embeds = np.array(list(map(spk_encoder.embed_utterance, speaker_wavs)))

    spk_emb = np.mean(utterance_embeds, axis=0)

    return spk_emb