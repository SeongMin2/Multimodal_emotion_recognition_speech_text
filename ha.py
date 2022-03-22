import torch
from torchtext import data
from torchtext import datasets

TEXT = data.Field(pad_first=True, fix_length=500)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

pass