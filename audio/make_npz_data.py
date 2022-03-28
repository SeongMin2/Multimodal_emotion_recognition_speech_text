import os, sys
import numpy as np
from pathlib import Path

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) # 상위 폴더 import 접근법
import parser_helper as helper
from audio.identity.resemblyzer import encode as encode

from phoneme.gentle.get_data import get_content_list
# get_content_list 밑에서 phone 추출에 사용하기

NUM_UTTRS = 2

def get_data(config):
    dir_name, subdir_list, _ = next(os.walk(config.spec_dir))
    helper.logger("info", "[INFO] Found directory: " + str(dir_name))
    
    for session in sorted(subdir_list): # 여기서 영숫순인듯
        helper.logger("info", "[INFO] Processing session")

        session_dir, gender_dir, _ = next(os.walk(os.path.join(dir_name,session)))

        for gender in gender_dir:
            # phone_seq extraction code needs to be inserted on here

            if(True): # Whether specturm files exist
                spk_id = session + "_" + gender

                helper.logger("info", f"[INFO] Proceesing speaker{spk_id}")
                print("spk: ", spk_id)
                _, _, wav_list = next(os.walk(os.path.join(session_dir,gender)))

                wav_idx = np.random.choice(len(wav_list),size=NUM_UTTRS, replace=False)
                helper.logger("info","[INFO] idx_uttrs: " + str(wav_idx))
                embeds = encode.get_speaker_embeddings(dir_name = config.wav_dir,
                                                       speaker = session + "/" + gender,
                                                       wav_list = wav_list,
                                                       idx_uttrs = wav_idx,
                                                       num_uttrs = NUM_UTTRS)

                # resemblyzer 이용하여 encode 결과 return한 부분 까지
                helper.logger("info", "[INFO] Extract speaker " +spk_id+ "speaker identity")

                # phone info 추출
















def main():
    config = helper.get_config()
    get_data(config)

if __name__ == '__main__':
    main()