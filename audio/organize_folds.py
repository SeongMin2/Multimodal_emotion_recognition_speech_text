"""
    Name: organize_folds.py
    Description: Organize the spmel directories in the 5 folds
"""

import shutil
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import parser_helper as helper

def main(config):

    # 이거 나중에 parser_helper에 갖다 놓을까..
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--spmel_dir", type=str, default="../audio/spectrum", help="directory of the spmel files")
    parser.add_argument("--folds_dir", type=str, default="../full_data/folds" ,help="directory to store the folds")

    config = parser.parse_args()
    '''

    src_dir = config.spec_dir
    folds_dir = config.folds_dir

    for session in range(1, 6):
        for fold in range(1, 6):
            if fold == session:
                shutil.copytree(src_dir + "/Session" + str(session), folds_dir + "/fold" + str(fold) + "/test/Session" + str(session))
            else:
                shutil.copytree(src_dir + "/Session" + str(session), folds_dir + "/fold" + str(fold) + "/train/Session" + str(session))

if __name__ == '__main__':
    config = helper.get_config()
    main(config)
