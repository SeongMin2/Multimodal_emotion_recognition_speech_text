import os, sys
import numpy as np

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) # 상위 폴더 import 접근법
import parser_helper as helper
# 이거 역시 문제 없이 잘 됨
from audio.identity.resemblyzer import encode

from phoneme.gentle.get_data import get_content_list
# get_content_list 밑에서 phone 추출에 사용하기

NUM_UTTRS = 8

def get_data(config):
    dir_name, subdir_list, _ = next(os.walk(config.spec_dir))
    helper.logger("info", "[INFO] Found directory: " + str(dir_name))

    utterances = []
    
    for session in sorted(subdir_list): # 여기서 영숫순인듯
        helper.logger("info", "[INFO] Processing session")

        session_dir, gender_dir, _ = next(os.walk(os.path.join(dir_name,session)))

        for gender in gender_dir:
            # phone_seq extraction code needs to be inserted on here

            if(True): # Whether specturm files exist
                spk_id = session + "_" + gender
                ############################################
                #           get speaker identity           #
                ############################################
                helper.logger("info", f"[INFO] Proceesing speaker{spk_id}")
                print("spk: ", spk_id)
                _, _, wav_list = next(os.walk(os.path.join(session_dir,gender)))

                wav_idx = np.random.choice(len(wav_list),size=NUM_UTTRS, replace=False)
                helper.logger("info","[INFO] idx_uttrs: " + str(wav_idx))
                spk_embs = encode.get_speaker_embeddings(dir_name = config.wav_dir,
                                                         speaker = session + "/" + gender,
                                                         wav_list = wav_list,
                                                         idx_uttrs = wav_idx,
                                                         num_uttrs = NUM_UTTRS)
                # spk_embs의 shape는 (256,) 맞음
                # 그리고 model forward 하는 부분에서 96만큼 expand 시킴

                # resemblyzer 이용하여 encode 결과 return한 부분 까지
                helper.logger("info", "[INFO] Extract speaker " +spk_id+ "speaker identity")


                # phone info 추출
                content_list = []
                content_list, success_cases, unsuccess_cases = get_content_list(file_list=wav_list,
                                                                                speaker_dir=session + "/" + gender,
                                                                                config=config)
                # get_content_list는 make_data_helper.py 안에 있고 이 get_content_list안에 phone_seq.py안에 있는 get_phone_seq라는 함수를 통해서
                # phone 정보 추출해서 리턴 되것지
                # content_list
                # 이미 여기서 text 처리 해버리는데

                if (len(content_list) > 0):
                    for element in content_list:
                        tmp_element = element.copy()
                        if (gender == "Female"):
                            gender_class = 0
                        elif (gender == "Male"):
                            gender_class = 1
                        else:
                            print("Error. Undefined gender " + str(gender))
                            sys.exit(1)
                        tmp_element.append(gender_class)
                        tmp_element.append(spk_embs)
                        tmp_element.append(spk_id) # 이 아이디를 저장할 필요가 있나 나는 어차피 파일 명 기준으로 할텐데 일단 냅두자
                        utterances.append(tmp_element)
                        # helper.logger("info", "[INFO] Utterances length: " + str(len(utterances)))
                else:
                    helper.logger("warning", "[WARNING] The speaker does not have any phone sequence")

    helper.save_data(utterances,config)  # make_data_helper.py에 있는 함수로 id, speaker identify, phone 정보 여기서 통째로 정리하네
    '''
    helper.logger("info", "[INFO] Total number of successful spec transformations: " + str(success_cases))
    helper.logger("info", "[INFO] Total number of unsuccessful spec transformations: " + str(unsuccess_cases))
    '''

def main():
    config = helper.get_config()
    get_data(config)

if __name__ == '__main__':
    main()