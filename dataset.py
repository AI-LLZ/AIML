import torch
import torchaudio
import os
import sys

from torch import nn  
from typing import List, Dict
from torch.utils.data import Dataset
from collections import defaultdict

import re
import torch
import glob

import pandas as pd
from utils import random_subsample

class CoswaraDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        audio_path: str,
        label_mapping: str,
        max_len: int = 96000,
        mode: str = "train"
    ):
        self.data = pd.read_csv(csv_path)
        self.audio_dir = audio_path
        with open(label_mapping) as f:
            self.label_mapping = eval(f.read())
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode
        self.padding = nn.ConstantPad1d(self.max_len, 0.0)

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
        ret = defaultdict(list)        
        for sample in samples:
            _id, _label = sample['id'], sample['label']
            root_path = os.path.join(self.audio_dir, _id, "*.wav")
            for wav_path in glob.glob(root_path):
                ret['id'].append(_id)
                ret['label'].append(_label)
                wav, sr = torchaudio.load(wav_path)
                # print(wav_path)
                if sr != 16000:
                    downsample = torchaudio.transforms.Resample(sr) # downsample to 16000 Hz
                    wav = downsample(wav)
                # print("original:",wav.shape)
                wav = random_subsample(wav, self.max_len)
                # print("after:", wav.shape)
                # print('+'*20)
                ret['wav'].append(wav)
        ret['label'] = torch.tensor(ret['label']).long()
        ret['wav'] = torch.stack(ret['wav'])
        return ret

if __name__ == "__main__":
    print(str(torchaudio.get_audio_backend()))
    from torch.utils.data import DataLoader
    dataset = CoswaraDataset(
        csv_path = "/tmp2/b08902001/coswara/combined_data.csv",
        audio_path = "/tmp2/b08902001/coswara",
        label_mapping = "/tmp2/b08902001/coswara/mapping.json",
    )
    train_loader = DataLoader(dataset, batch_size=8,
                            shuffle=False, collate_fn=dataset.collate_fn)

    for i, samples in enumerate(train_loader):
        if i: break
        print(samples['id'])
        print(samples['label'].shape)
        print(samples['wav'].shape)