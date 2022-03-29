from torch.utils import data
import torch
import numpy as np
import pickle
import os, math, sys

# 생각해보니까 make_spec할 때 spectrum 잘 만들어주는데 collate_fn이 필요 없지 않나..
# 그렇다면 make_spect는 audio마다 길이가 다른데 어떻게 처리한지 한 번 확인해봐야 할듯
# 확인해보니까 spec shape 다 다름 따라서 collate_fn 해야함
# def get_data_loader():
