import numpy as np
import os, sys
from pathlib import Path
import phoneme.gentle.phone_seq as ph
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import parser_helper as helper
# 여기서 emotion class 추출 안할거임
# 그리고 그거 수정해라 그 txt dir 추출하는게 아니라 그냥 text data 전달해서 할거임

def is_file_unk_token(txt_f):
    """ Identify if the txt file is only an unknown token. """
    # read the txt file
    with open(txt_f, "r") as txt_p:
        transcript = txt_p.readline()
    transcript = transcript.strip()
    if((transcript[0] == "[") and (transcript[-1] == "]")):
        return True
    else:
        return False


def get_content_list(file_list, speaker_dir, config):
    """
        Get the content list for a given speaker
        The content list includes the utterance file path,
        the sequence of phones and durations, the sequence
        of main phones and the emotion for each utterance
    """
    content_list = []
    unsuccess_cases = 0
    success_cases = 0

    # for all files of a given speaker
    for file_name in sorted(file_list):
        print("Processing ", file_name)
        json_file_name = file_name[:-4] + '.json'   # 애초에 이 파일을 사용한다는것은 기존에 미리 extract_phonemes로 추출해서 저장된 상태에서 실행되어져야 함
        phone_seq_file_path = config.phone_dir + '/' + speaker_dir + '/' + json_file_name
        pathlib_phone_path = Path(phone_seq_file_path)

        spec = np.load(str(config.spec_dir) + '/' + speaker_dir + '/' + file_name) # load spectrogram

        if pathlib_phone_path.exists():
            # The speech has phonetic content
            phones_and_durations, main_phones, success = ph.get_phone_seq(json_file=phone_seq_file_path,
                                                                          config=config,
                                                                          spec_frames=spec.shape[0], # 그냥 spec.shape[0]의 shape임
                                                                          speaker_dir=speaker_dir,
                                                                          file_name=file_name)
            # 여기에 있구나 get_phone_seq을 통해 phone 정보 추출
            # 여기서는 spectrum의 정보가 사전에 필요한것 같음

            if success:
                assert main_phones.shape[0] == spec.shape[0]
                utt_file = os.path.join(speaker_dir ,file_name)
                word_seq, word_intervals = ph.get_word_seq_and_intervals(json_file=phone_seq_file_path)
                utt_content = [str(utt_file), phones_and_durations, main_phones, word_seq, word_intervals] # save spmel file name and phone sequence
                content_list.append(utt_content)
                success_cases += 1
            else:
                unsuccess_cases += 1
        else:
            # The speech corresponds to a "silent" speech get the transcript file
            txt_f = config.txt_dir + '/' + speaker_dir + '/' + file_name[:-4] + '.txt'
            txt_f_pt = Path(txt_f)
            if txt_f_pt.exists():
                # if the file only has unknown tokens (no phone information)
                if is_file_unk_token(txt_f):
                    phones_and_durations, main_phones = ph.get_silent_phone_seq(config=config,
                                                                                spec_frames=spec.shape[0],
                                                                                speaker_dir=speaker_dir,
                                                                                file_name=file_name)
                    assert main_phones.shape[0] == spec.shape[0]
                    utt_file = os.path.join(speaker_dir ,file_name)
                    utt_content = [str(utt_file), phones_and_durations, main_phones, [None], [None]] # save spmel file name and phone sequence
                    content_list.append(utt_content)
                    success_cases += 1
                else:
                    helper.logger("warning", "[WARNING] File " + str(txt_f) + " was not aligned properly!")
                    unsuccess_cases =+ 1
            else:
                helper.logger("warning", "[WARNING] Unexisting path " + str(pathlib_phone_path) + " or " + str(txt_f))
                unsuccess_cases = + 1

    helper.logger("info", "[INFO] Number of successful specs: " + str(success_cases))
    helper.logger("info", "[INFO] Number of unsuccessful specs: " + str(unsuccess_cases))
    return content_list, success_cases, unsuccess_cases
