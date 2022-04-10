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
# from ABS_PATH import ABS_PATH

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
# 위는 window일 경우임
# 리눅스의 경우는 다음과 같이 진행함
# ABS_PATH = os.path.dirname(os.path.abspath(__file__)).rsplit("/", 1)[0]
# 아님 그럴필요 없음

now= datetime.now()
date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
LOG_PATH = date_time + ".log"

ABS_PATH = ABS_PATH

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
    parser.add_argument("--md_save_dir", type=str, default=ABS_PATH + "/model/save", help="Path to Model save dir")

    parser.add_argument("--batch_size" ,type=int, default=2, help="Batch size")
    parser.add_argument("--dropout_ratio", type=float, default= 0.2, help="Ratio of dropout")
    parser.add_argument("--attention_emb", type=int , default=128, help="Size of attention hidden states")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of Multi-head")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning ratio")
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--log_interval", type=int, default=200, help="Interval time where model checks probability")

    # Bottleneck configuration
    parser.add_argument("-dim_neck", type=int, default=8, help="Bottleneck parameter of d")
    parser.add_argument("--freq", type=int, default=48, help="Bottleneck parameter of f : sampling frequency")


    # Input spectrogram configuration
    parser.add_argument("--len_crop", type=int, default=96, help="Dataloader output sequence length")
    parser.add_argument("--num_mels", type=int, default=80, help="Number of mel features at each frame (it should include delta and delta-delta if using those features)")
    parser.add_argument("--dim_wav2vec_emb", type=int, default=1024, help="Number of wav2vec features length")
    parser.add_argument("--speech_input", type=str, default="wav2vec", help="Encoding method of speech input")

    parser.add_argument("--dim_spk_emb", type=int, default=256, help="number of speaker identity embedding length")
    parser.add_argument("--dim_phone_emb", type=int, default=128, help="Dimension size of phoneme (Number of phoneme embedding length)")

    # model feauter size configuration
    parser.add_argument("--conv_dim", type=int, default=512, help="Number of convolution channel or Number of kernels")
    # Paper에는 2048로 나와 있는데 code는 512를 쓰네 ㅎ
    parser.add_argument("--dim_pre", type=int, default=512, help="Length of first LSTM module in Decoder")

    parser.add_argument("--txt_feat_model", type=str, default="bert-base-uncased", help='Text model for feature extraction')
    parser.add_argument("--max_token_len", type=int, default=124, help="max text sequence length")

    parser.add_argument("--selected_catg", type=list, default=["hap", "exc", "ang", "sad", "neu"], help="Limit categories")
    parser.add_argument("--n_classes", type=int , default=4, help="Number of classes(categories)")

    parser.add_argument("--pretrained_model", type=str, default=None, help="Path of pretrained model")


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
    elif level_name == 'training':
        logging.info(message)
        if show_terminal:
            print(colored(message, 'blue'))
    elif level_name == 'error':
        logging.error(message)
        if show_terminal:
            print(colored(message, 'red'))
    elif level_name == 'warning':
        logging.warning(message)
        if show_terminal:
            print(colored(message,'yellow'))