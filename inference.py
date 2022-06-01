import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import numpy as np
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

    model = M5(1,2, act_fn="Sigmoid") # change if you want to use different model
    state_dict = torch.load(args.model_path) # load the trained model
    model.load_state_dict(state_dict['model'])

    dataset = CoswaraDataset(
        csv_path = os.path.join(args.data_dir, "combined_data.csv"),
        audio_path = args.data_dir,
        label_mapping = os.path.join(args.data_dir, "mapping.json"),
        max_len = args.max_len
    )
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate_fn)

    model, data_loader = accelerator.prepare(model, data_loader) 

    embeddings, labels = validate(data_loader, model)
    
    # clf = LogisticRegression(solver='liblinear', random_state=1126, max_iter=100).fit(embeddings, labels)
    print(labels)
    clf = XGBClassifier(objective="binary:logistic", random_state=1126)
    clf.fit(embeddings, labels)
    print("* Train result:", clf.score(embeddings, labels), "auc", roc_auc_score(np.eye(3)[labels], clf.predict_proba(embeddings)))
    gt = args.gt

    wav, sr = torchaudio.load(args.wav_path, normalize = True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = random_subsample(wav, args.max_len)

    wav = wav.unsqueeze(0) # so that it has the shape (batch_size, ...)
    wav = wav.to(accelerator.device)
    with torch.no_grad():
        embeddings = model.get_embedding(wav).detach().cpu().numpy()

    print(clf.predict(embeddings))

    # acc = (logits.argmax(dim=-1) == 0).cpu().float().mean()
    # auc = acc # roc_auc_score(np.array([gt]), logits.detach().cpu().numpy()[:,1])

    # print("Accuracy:", acc, "AUC:", auc)
    
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
