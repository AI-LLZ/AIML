import torch
import numpy as np
import random
from torch import nn

from random import randint
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def random_subsample(wav, max_length):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    # print('called this (', wav.shape, max_length, end = ')-> ')
    n = wav.shape[1]
    if n < max_length: # padding 
        # print("early return => because n =", n, end = ' ')
        wav = torch.cat([wav, torch.zeros(max_length - n).unsqueeze(0)], dim=1)
        return wav
    # random truncation
    random_offset = randint(0, n - max_length - 1)
    # print(random_offset, end = '=> ')
    return wav[:, random_offset : random_offset + max_length]

