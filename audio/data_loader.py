from torch.utils import data
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os, math, sys
from audio.dataset import SpeechTextDataset
import parser_helper as helper

# 생각해보니까 make_spec할 때 spectrum 잘 만들어주는데 collate_fn이 필요 없지 않나..
# 그렇다면 make_spect는 audio마다 길이가 다른데 어떻게 처리한지 한 번 확인해봐야 할듯
# 확인해보니까 spec shape 다 다름 따라서 collate_fn 해야함
def get_data_loaders(config, train_npz, test_npz, num_workers=0):

    train_dir = config.train_dir
    test_dir = config.test_dir
    wav_dir = config.wav_dir
    batch_size = config.batch_size

    helper.logger("info", "[INFO] Data loading...")

    train_dataset = SpeechTextDataset(config,"train",train_dir, wav_dir, train_npz, 16000)
    # a = train_dataset[0]

    test_dataset = SpeechTextDataset(config, "test", test_dir, wav_dir, test_npz, 16000)

    # get the evaluation datasets (to make utterance-level evaluations from the segments)
    train_eval_dataset = SpeechTextDataset(config, "test", train_dir, wav_dir, train_npz, 16000)

    # get the train dataset
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   drop_last=True,
                                   worker_init_fn=worker_init_fn,
                                   pin_memory=True)

    # get the test dataset cut to perform the utterance-level prediction
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=True)

    # get the train dataset cut to perform the utterance-level prediction
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    train_eval = data.DataLoader(dataset=train_eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 drop_last=False,
                                 worker_init_fn=worker_init_fn,
                                 pin_memory=True)
    '''
    # training dataset with batch size = 1
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    train_1batch = data.DataLoader(dataset=train_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   # num_workers=num_workers,
                                   drop_last=False,
                                   worker_init_fn=worker_init_fn,
                                   pin_memory=True)
    '''

    return train_loader, test_loader, train_eval


def get_train_data_loaders(config, train_npz, test_npz, num_workers=4):

    train_dir = config.train_dir
    test_dir = config.test_dir
    wav_dir = config.wav_dir
    batch_size = config.batch_size

    helper.logger("info", "[INFO] Data loading...")

    train_dataset = SpeechTextDataset(config,"train",train_dir, wav_dir, train_npz, 16000)
    test_dataset = SpeechTextDataset(config, "test", test_dir, wav_dir, test_npz, 16000)

    # a = train_dataset[0]

    #result = train_dataset[0]
    #result2 = train_dataset[1]
    #a = train_dataset[2]

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size = batch_size,
                              shuffle=True,
                              # num_workers = num_workers, # num_workers를 추가하면 iter()이 못읽음 ...
                              worker_init_fn=worker_init_fn,
                              drop_last=True
                              )

    # get the test dataset cut to perform the utterance-level prediction
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  # num_workers=num_workers,
                                  drop_last=False,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=True)
    '''
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   drop_last=True,
                                   worker_init_fn=worker_init_fn,
                                   pin_memory=True)
    '''

    return train_loader, test_loader







