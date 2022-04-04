# import sys
# from os.path import dirname, join, abspath
# sys.path.insert(0, abspath(join(dirname(__file__), '')))
from audio.data_loader import get_data_loader
import parser_helper as helper

from torch.backends import cudnn

# Name of the train and test pkl files
train_npz = "train"
test_npz = "test"



def main():
    config = helper.get_training_config()
    get_data_loader(config, train_npz, test_npz)

if __name__ == '__main__':
    main()