import re
import os
import pandas as pd

PWD_PATH = './'
SAVE_PATH = './extraction/'

SAVE_TO_CSV = False

info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

text_ext = re.compile('(?<=]: )(.*?)(?=\n)')  # 앞 뒤 둘 다 미포함
key_ext = re.compile('(?=Ses)(.*?)(?= )')  # 앞은 포함 뒤는 미포함


def extract_data(save_path, sess_list) -> pd.DataFrame: # 이게 리턴 주석 맞은지 추후에 확인

    df = pd.DataFrame(columns=['wav_file','text', 'emotion'])

    for sess in sess_list:
        wav_file_names, emotions = [], []
        wav_file_names_2, text = [], []

        emo_evaluation_dir = PWD_PATH + 'IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]

        for file in evaluation_files:
            with open(emo_evaluation_dir + file) as f:
                content = f.read()

            info_lines = re.findall(info_line, content)

            for line in info_lines[1:]:  # the first line is a header
                wav_file_name, emotion= line.strip().split('\t')[1:3]
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)

        text_dir = PWD_PATH + 'IEMOCAP_full_release/Session{}/dialog/transcriptions/'.format(sess)
        text_files = [l for l in os.listdir(text_dir)]

        for file in text_files:
            with open(text_dir + file) as f:
                content = f.read()

            keys = re.findall(key_ext, content)
            utterances = re.findall(text_ext, content)

            for key in keys:
                wav_file_names_2.append(key)

            for utr in utterances:
                text.append(utr)

        df_emoeval = pd.DataFrame(columns=['wav_file', 'emotion'])
        df_text = pd.DataFrame(columns=['wav_file', 'text'])

        df_emoeval['wav_file'] = wav_file_names
        df_emoeval['emotion'] = emotions

        df_text['wav_file'] = wav_file_names_2
        df_text['text'] = text

        df_iemocap = pd.merge(left=df_text, right=df_emoeval, how="inner", on="wav_file")

        if(SAVE_TO_CSV):
            df_iemocap.to_csv(save_path + 'df_v2_iemocap_sess{}.csv'.format(sess), index=False)

        df = pd.concat([df, df_iemocap])

    df.reset_index(inplace=True, drop=True)

    return df


def get_data(sess_list, save_path):
    wav_paths = list()
    transcripts = list()
    emotions = list()

    df = extract_data(sess_list, save_path)
    for idx in range(len(df)):
        wav_paths.append(df['wav_file'][idx])
        transcripts.append(df['text'][idx])
        emotions.append(df['emotion'][idx])

    return wav_paths, transcripts, emotions


def main():
    sess_list = [1]
    save_path = './extraction/'

    wav_paths, transcripts, emotions = get_data(save_path,sess_list) # list들이 return 됨
    pass

if __name__ == '__main__':
    main()