import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from audio.data_loader import get_data_loader
from parser_helper import get_config

from torch.backends import cudnn

# Name of the train and test pkl files
train_npz = "train"
test_npz = "test"