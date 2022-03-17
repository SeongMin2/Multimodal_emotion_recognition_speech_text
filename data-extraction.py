import re
import os
import pandas as pd

PWD_PATH = './'
SAVE_PATH = './extraction/'

info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

text_ext = re.compile('(?<=]: )(.*?)(?=\n)')  # 앞 뒤 둘 다 미포함
key_ext = re.compile('(?=Ses)(.*?)(?= )')  # 앞은 포함 뒤는 미포함

# start_times, end_times, wav_file_names, wav_file_names_2, text, emotions, vals, acts, doms = [], [], [], [], [], [], [], [], []

for sess in [1,2,3,4,5]:
    start_times, end_times, wav_file_names, wav_file_names_2, text, emotions, vals, acts, doms = [], [], [], [], [], [], [], [], []

    emo_evaluation_dir = PWD_PATH + 'IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
    evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]

    for file in evaluation_files:
        with open(emo_evaluation_dir + file) as f:
            content = f.read()

        info_lines = re.findall(info_line, content)

        for line in info_lines[1:]:  # the first line is a header
            start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
            start_time, end_time = start_end_time[1:-1].split('-')
            val, act, dom = val_act_dom[1:-1].split(',')
            val, act, dom = float(val), float(act), float(dom)
            start_time, end_time = float(start_time), float(end_time)

            start_times.append(start_time)
            end_times.append(end_time)
            wav_file_names.append(wav_file_name)
            emotions.append(emotion)
            vals.append(val)
            acts.append(act)
            doms.append(dom)

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

    df_emoeval = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])
    df_text = pd.DataFrame(columns=['wav_file', 'text'])

    df_emoeval['start_time'] = start_times
    df_emoeval['end_time'] = end_times
    df_emoeval['wav_file'] = wav_file_names
    df_emoeval['emotion'] = emotions
    df_emoeval['val'] = vals
    df_emoeval['act'] = acts
    df_emoeval['dom'] = doms

    df_text['wav_file'] = wav_file_names_2
    df_text['text'] = text

    df_iemocap = pd.merge(left=df_emoeval, right=df_text, how="inner", on="wav_file")

    df_iemocap.to_csv(SAVE_PATH + 'df_iemocap_sess{}.csv'.format(sess), index=False)