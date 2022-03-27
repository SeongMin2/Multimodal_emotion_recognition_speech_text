import argparse
import logging
from datetime import datetime
from termcolor import colored

now= datetime.now()
date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
LOG_PATH = date_time + ".log"

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--spec_dir", type=str, default="./spectrum", help="Path to spectrum files")
    parser.add_argument("--ph_dict_dir", type=str, default="../phoneme/phone_dict.csv",help="path to phone dictionary file")
    parser.add_argument("--wav_dir", type=str, default="../data/speech", help="Path to wave files")

    parser_config = parser.parse_args()

    return parser_config

# 이런것도 있넿 활용성 좋은듯
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

