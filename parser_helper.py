import argparse
import logging
from datetime import datetime
from termcolor import colored
import os
import numpy as np
from pathlib import Path
from phoneme.gentle import phone_seq as ph
import glob
import pandas as pd

now= datetime.now()
date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
LOG_PATH = date_time + ".log"

ABS_PATH = 'C:\SPB_Data\iemocap_preprocessing'

def get_label(config, file_name, session):
    file_name = file_name[:-4]
    tables = glob.glob(config.table_dir + "/" + "*.csv")
    table_name = [file for file in tables if session in file]
    table_path = table_name[0]
    df = pd.read_csv(table_path)

    label = df[df['wav_file'] == file_name]['emotion'].values[0]

    return label

def save_data(speakers, config):
    """ Save npz data file """
    logger('info','[INFO] Speakers length: ' + str(len(speakers)))

    np.savez(config.npz_dir+'/' +config.mode+'.npz', feature=speakers) # 내가 feature라고 설정한거임
# 일단 이렇게 해놓고 나중에 수정

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="test", help="train or test mode")
    parser.add_argument("--spec_dir", type=str, default=ABS_PATH + "/full_data/folds/fold5/test", help="Path to spectrum files")
    # parser.add_argument("--spec_dir", type=str, default=ABS_PATH + "/audio/spectrum", help="Path to spectrum files")
    parser.add_argument("--phone_dir", type=str, default=ABS_PATH + "/phoneme/gentle/align_results", help="path to the phonetic alignment json files")
    parser.add_argument("--phone_dict_file", type=str, default=ABS_PATH + "/phoneme/gentle/phone_dict.csv",help="path to phone dictionary file")
    parser.add_argument("--wav_dir", type=str, default=ABS_PATH + "/full_data/speech", help="Path to wave files")
    # parser.add_argument("--npz_dir", type=str, default=ABS_PATH + "/full_data", help="Path to npz file")
    parser.add_argument("--freq", type=int, default=16000, help="speech frequency")
    parser.add_argument("--hop_len", type=int, default=320, help="hop length")
    parser.add_argument("--folds_dir", type=str, default=ABS_PATH + "/full_data/folds", help="/audio/organize_folds.py")
    parser.add_argument("--table_dir", type=str, default=ABS_PATH + "/full_data/table", help="Path to table dir")

    # 이렇게 parser_helper가 최상위에 있으면 parser_helper 기준과 해당 parameter을 실질적으로 사용하는 코드의 위치가 달라져 버림
    # 그래서 이런 parser_helper은 사용하고자 하는 파일과 동일한 위치에 넣어줘야하는것이 맞음
    # 노우 그렇지 않음 -> 절대 경로로 설정하면 이런일 없음

    parser_config = parser.parse_args()

    phone_dict = ph.get_phone_dict(parser_config.phone_dict_file)

    parser_config.npz_dir = str(parser_config.spec_dir) # 여기에 이 spectrum의 경로를 fold 경로로 잡게 되어서 ㅇㅇ 위에서는 default이고 저기에 입력 넣어줄거임
    parser_config.frame_len = float(parser_config.hop_len / parser_config.freq)  # = hop_length/freq spectrogram frame duration in seconds # frame의 길이를 초로 표현
    parser_config.phone_dict = phone_dict


    return parser_config

def get_training_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./full_data", help="Path to data dir")
    parser.add_argument("--train_dir", type=str, default=ABS_PATH + "/full_data/folds/fold1/train", help="Path to train data dir")
    parser.add_argument("--test_dir", type=str, default= ABS_PATH + "/full_data/folds/fold1/test", help="Path to test data dir")
    parser.add_argument("--wav_dir", type=str, default=ABS_PATH + "/full_data/speech", help="Path to wave files")

    parser.add_argument("--speech_input", type=str, default="wav2vec", help="Encoding method of speech input")

    parser.add_argument("--batch_size" ,type=int, default=2, help="Batch size")

    # Input spectrogram configuration
    parser.add_argument("--len_crop", type=int, default=96, help="dataloader output sequence length")
    parser.add_argument("--num_mels", type=int, default=80, help="number of mel features at each frame (it should include delta and delta-delta if using those features)")
    parser.add_argument("--wav2vec_feat_len", type=int, default=1024, help="size of wav2vec features")


    parser_config = parser.parse_args()

    return parser_config


def logger(level_name, message, log_path=LOG_PATH, highlight=False, show_terminal=True):
    """
        Write message to the log
        Input:
                level_name : level name (e.g., error, warning, etc)
                message    : message to be printed in the screen (prompt) and log
                log_path   : path of the log file
                highlight  : boolean indicating if the message should be highlighted in the prompt
        Output: void
    """
    # log configuration
    logging.basicConfig(filename=log_path, filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')

    if level_name == 'info':
        logging.info(message)
        if show_terminal:
            if highlight:
                print(colored(message,'white', 'on_green'))
            else:
                print(colored(message,'green'))
    elif level_name == 'error':
        logging.error(message)
        if show_terminal:
            print(colored(message,'red'))
    elif level_name == 'warning':
        logging.warning(message)
        if show_terminal:
            print(colored(message,'yellow'))