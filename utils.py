import torch
import numpy as np
import random
from torch import nn
from torchaudio import transforms

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

def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):

    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(16000, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(aud)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec
