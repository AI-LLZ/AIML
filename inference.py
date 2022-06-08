import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from regex import W

import torch
import numpy as np
from transformers import get_constant_schedule
import torchaudio
from accelerate import Accelerator
from torch import logit, nn
import sys
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from dataset import CoswaraDataset
from models import M5, M18 
from utils import random_subsample, same_seeds
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
import timm 
from lightgbm import LGBMClassifier
import torch.nn.functional as F



N_SOUND = 9

@torch.no_grad()
def validate(data_loader, model):
    model.eval()
    valid_loss = []
    valid_accs = []
    all_labels, all_embeddings = [], []

    for idx, batch in enumerate(tqdm(data_loader)):
        logits = model.get_embedding(batch['wav'])
        labels = batch['label']

        all_labels.extend(labels.cpu().numpy())
        if not len(all_embeddings):
            all_embeddings = logits.detach().cpu().numpy()
        else:
            all_embeddings = np.concatenate([all_embeddings, logits.detach().cpu().numpy()])

    all_labels = np.array(all_labels)
    return all_embeddings, all_labels



def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator()

    cough_model = timm.create_model('resnet34d', num_classes=2, pretrained=True, in_chans=1)
    breathe_model = timm.create_model('resnet34d', num_classes=2, pretrained=True, in_chans=1) 
    status_model = LGBMClassifier(n_estimators=32, learning_rate=0.01, num_leaves=32, max_depth=4)


    cough_stat_dict = torch.load(os.path.join(args.ckpt_dir, 'ckpt-cough/_best.ckpt'))
    breathe_state_dict = torch.load(os.path.join(args.ckpt_dir, 'ckpt-breathing/_best.ckpt'))

    cough_model.load_state_dict(cough_stat_dict['model'])
    breathe_model.load_state_dict(breathe_state_dict['model'])
    with open(f'lgbm.pkl', 'rb') as f:
        status_model = pickle.load(f)

    cough_model, breathe_model = accelerator.prepare(cough_model, breathe_model)

    cough_model.eval()
    breathe_model.eval()

    cough = get_cough()
    breathe = get_breathe()
    status = get_status()

    cough_logits = cough_model(cough)
    cough_pred = F.softmax(torch.argmax(cough_logits, dim=1), dim=1)
    breathe_logits = breathe_model(breathe)
    breathe_pred = F.softmax(torch.argmax(breathe_logits, dim=1), dim=1)
    status_pred = status_model.predict_proba(status)
    def customEye(N : int, diagVal : float, otherVal : float):
        a = np.full((N, N), otherVal)
        a[np.diag_indices(N)] = diagVal
        return a
    

    ensemble_test_pred = (breathe_pred @ customEye(2, 0.45, 0.)) + (cough_pred @ customEye(2, 0.45, 0.)) + (status_pred @ customEye(2, 0.1, 0.))

    print(np.argmax(ensemble_test_pred, axis=-1))
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=5920)
    # data
    parser.add_argument("--data_dir", type=str, default="./coswara")
    parser.add_argument("--max_len", type=int, default=96000)
    parser.add_argument("--wav_path", type=str, default="data/test/wav/0.wav")
    parser.add_argument("--gt", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="models/m5_0.pth")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
