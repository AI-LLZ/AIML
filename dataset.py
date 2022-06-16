from random import random
import torch
import torchaudio
import os
import sys

from torch import nn  
from typing import List, Dict
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np

import re
import torch
import glob

import pandas as pd
from utils import random_subsample
from torchaudio import transforms


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

all_mean = -1.0131372e-05
all_std = 0.06298088

class CoswaraDataset(Dataset):
    def __init__(
        self,
        train_csv: pd.DataFrame,
        audio_path: str,
        label_mapping: str,
        audio_type: str,
        max_len: int = 160000,
        squeeze = False,
        mode: str = "train",
        return_type = "pt"
    ):
        #df = pd.read_csv(csv_path)
        self.audio_dir = audio_path
        with open(label_mapping) as f:
            self.label_mapping = eval(f.read())
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        #self.mode = mode
        self.padding = nn.ConstantPad1d(self.max_len, 0.0)
        #self.regression_mapping = [0, 0, 1, 1, 1, 0, 0]
        self.squeeze = squeeze
        #n_samples = 100000
        #df['covid_status'] = df['covid_status'].map(lambda x: self.regression_mapping[x])
        #n_labels = len(set(df['covid_status']))
        #for i in range(n_labels):
        #    n_samples = min(n_samples, len(df[df['covid_status'] == i]))
        #print(f"{n_samples}, {n_labels}")
        #df = df.groupby('covid_status').apply(lambda x: x.sample(n_samples, random_state=42))
        self.data = train_csv
        self.audio_type = audio_type
        #self.return_type = return_type
        print(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data.iloc[index]
        ret = { "id": instance.id, "label": instance.covid_status }
        return ret

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

    def collate_fn(self, samples: List[Dict]) -> Dict:
        ret = {'id': [], 'label': [], 'wav': []}
        for sample in samples:
            _id, _label = sample['id'], sample['label']
            root_path = os.path.join(self.audio_dir, _id, f"{self.audio_type}-*.wav")
            tmp_wav = []
            for wav_path in glob.glob(root_path):
                wav, sr = torchaudio.load(wav_path)
                if wav.shape[1] < 1600: 
                    wav = torch.zeros(1,16000)
                # print(wav_path)
                if sr != 16000:
                    downsample = torchaudio.transforms.Resample(sr) # downsample to 16000 Hz
                    wav = downsample(wav)
                # print("original:",wav.shape)
                


                wav = random_subsample(wav, self.max_len)
                wav = (wav - all_mean) / all_std
                #wav = wav.squeeze()
                #print(wav)
                sgram = spectro_gram(wav, n_mels=64, n_fft=1024, hop_len=None)
                aug_sgram = spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
            
                aug_sgram = torch.tensor(aug_sgram)
                if len(tmp_wav) == 1:
                    tmp_wav[0] = torch.cat((tmp_wav[0], aug_sgram), dim = 1)
                else:
                    tmp_wav.append(aug_sgram)
                # ret['wav'].append(aug_sgram)
                # ret['id'].append(_id)
                # ret['label'].append(_label) # self.regression_mapping[_label])

            if len(tmp_wav) != 0:
                shap = tmp_wav[0].shape

            if len(tmp_wav) != 0 and shap[1] == 128:
                #print("kasdl;fjklskdjkl"
                if len(ret['wav']) == 0:
                    ret['wav'] = tmp_wav[0]
                else:
                    ret['wav'] = torch.cat((ret['wav'], tmp_wav[0]))
                #ret['wav'].append(tmp_wav)
                ret['id'].append(_id)
                ret['label'].append(_label)
        
        ret['wav'] = torch.unsqueeze(ret['wav'], 1)

        ret['label'] = torch.tensor(ret['label']).long()
        #print(ret['label'].shape)
        #ret['wav'] = torch.stack(ret['wav'])
    
        #print(ret['wav'].shape)

        if self.squeeze:
            ret['wav'] = ret['wav'].squeeze(1)
        #if self.return_type == "np":
        #    ret['wav'] = ret['wav'].numpy()
        return ret

if __name__ == "__main__":
    print(str(torchaudio.get_audio_backend()))
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from tqdm import tqdm

    lens = []
    df = pd.read_csv('./coswara/combined_data.csv')
    for _id in df.id:
        for file in glob.glob(f'./coswara/{_id}/cough-*.wav'):
            wav, sr = torchaudio.load(file)
            if sr != 16000:
                print(file)
            lens.append(wav.shape[1])
    lens = np.array(lens)
    print(lens.max(), lens.min(), lens.mean())

    exit()

    accelerator = Accelerator()
    dataset = CoswaraDataset(
        csv_path = "./coswara_normalized/combined_data.csv",
        audio_path = "./coswara_normalized",
        label_mapping = "./coswara_normalized/mapping.json",
    )
    train_size = int(round(len(dataset) * 0.8))
    valid_size = len(dataset) - train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(
        train_set,
        collate_fn=train_set.dataset.collate_fn,
        shuffle=True,
        batch_size=2,
    )
    valid_loader = DataLoader(
        valid_set,
        collate_fn=valid_set.dataset.collate_fn,
        shuffle=False,
        batch_size=2
        )

    train_loader, valid_loader = accelerator.prepare(train_loader, valid_loader)
    for idx, batch in enumerate(tqdm(train_loader)):
        pass
    for idx, batch in enumerate(tqdm(valid_loader)):
        pass
        
