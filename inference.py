import io
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pyparsing import original_text_for

import torch
import numpy as np
from accelerate import Accelerator

from tqdm import tqdm
from dataset import CoswaraDataset
from models import M5, M18
from utils import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
import timm
from lightgbm import LGBMClassifier
import torch.nn.functional as F
import pickle

from flask import Flask, jsonify, request
import json
from flask_cors import CORS

import soundfile as sf
import torchaudio
import pandas as pd

N_SOUND = 9

feature_cols = [
    "label_ep", # english proficient
    "label_rU", # returning User
    "label_smoker",
    "label_cold",
    "label_ht", # Hypertension
    "label_diabetes",
    "label_cough",
    "label_ctScan",
    "label_ctScore",
    "label_diarrhoea",
    "label_fever",
    "label_loss_of_smell",
    "label_mp", # Muscle Pain
    "label_testType",
    "label_um",
    "label_vacc",
    "label_bd",
    "label_others_resp",
    "label_ftg",
    "label_st",
    "label_ihd", # Fatigue
    "label_asthma",
    "label_others_preexist",
    "label_cld", # Chronic Lung Disease
    "label_pneumonia",
]


@torch.no_grad()
def validate(data_loader, model):
    model.eval()
    valid_loss = []
    valid_accs = []
    all_labels, all_embeddings = [], []

    for idx, batch in enumerate(tqdm(data_loader)):
        logits = model.get_embedding(batch["wav"])
        labels = batch["label"]

        all_labels.extend(labels.cpu().numpy())
        if not len(all_embeddings):
            all_embeddings = logits.detach().cpu().numpy()
        else:
            all_embeddings = np.concatenate(
                [all_embeddings, logits.detach().cpu().numpy()]
            )

    all_labels = np.array(all_labels)
    return all_embeddings, all_labels


def predict(args, cough=None, breathe=None, status=None):
    same_seeds(args.seed)

    cough_model = timm.create_model(
        "resnet34d", num_classes=2, pretrained=True, in_chans=1
    )
    breathe_model = timm.create_model(
        "resnet34d", num_classes=2, pretrained=True, in_chans=1
    )
    # status_model = LGBMClassifier(
    #     n_estimators=32, learning_rate=0.01, num_leaves=32, max_depth=4
    # )

    # cough_stat_dict = torch.load(os.path.join(args.ckpt_dir, "ckpt-cough/_best.ckpt"))
    # breathe_state_dict = torch.load(
    #     os.path.join(args.ckpt_dir, "ckpt-breathing/_best.ckpt")
    # )

    # cough_model.load_state_dict(cough_stat_dict["model"])
    # breathe_model.load_state_dict(breathe_state_dict["model"])
    # with open(f"lgbm.pkl", "rb") as f:
    #     status_model = pickle.load(f)


    cough_model.eval()
    breathe_model.eval()

    cough_logits = cough_model(cough).detach()
    cough_pred = F.softmax(cough_logits, dim=-1)
    breathe_logits = breathe_model(breathe).detach()
    breathe_pred = F.softmax(breathe_logits, dim=-1)
    print(f"{cough_pred.shape=}, {breathe_pred.shape=}")

    def customEye(N: int, diagVal: float, otherVal: float):
        a = np.full((N, N), otherVal)
        a[np.diag_indices(N)] = diagVal
        return a

    ensemble_test_pred = (
        cough_pred + breathe_pred
    )

    prediction = np.argmax(ensemble_test_pred, axis=-1).item()
    print(prediction)
    return prediction


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=5920)
    # data
    args = parser.parse_args()
    return args

def process_audio(cough_heavy, cough_light, breathe_heavy, breathe_light):
    all_mean = -1.0131372e-05
    all_std = 0.06298088

    downsample = torchaudio.transforms.Resample(orig_freq=44100)

    tmp_list = [cough_heavy, cough_light, breathe_heavy, breathe_light]
    for idx in range(len(tmp_list)):
        try:
            bytes = tmp_list[idx]
            wav, _ = sf.read(file=io.BytesIO(bytes), dtype='float32')
            wav = np.expand_dims(wav, axis=0)
            print(wav.squeeze().shape)
            wav = downsample(torch.from_numpy(wav))
            wav = random_subsample(wav, 96000)

            wav = (wav - all_mean) / all_std

            sgram = spectro_gram(wav)
            aug_sgram = spectro_augment(sgram, n_time_masks=2)

            tmp_list[idx] = aug_sgram
        except Exception as e:
            print(e)

    cough = torch.cat((tmp_list[0], tmp_list[1]), dim=1).unsqueeze(0)
    breathe = torch.cat((tmp_list[2], tmp_list[3]), dim=1).unsqueeze(0)
    print(cough.shape, breathe.shape)
    return cough, breathe 


def main(args):
    app = Flask(__name__)
    CORS(app)

    @app.route("/submit", methods=["POST"])
    def submit():
        if request.method == "POST":
            print(request.form)
            try:
                cough_heavy = request.files.get('heavy_cough').read()
                cough_light = request.files.get('soft_cough').read()
                breathe_heavy = request.files.get('heavy_breath').read()
                breathe_light = request.files.get('soft_breath').read()
                cough, breathe = process_audio(cough_heavy, cough_light, breathe_heavy, breathe_light)
            except Exception as e:
                print(e)
                return "Error"


        return "positive" if predict(args, cough, breathe) else "negative"

    app.run(host="0.0.0.0", port=5920, debug=True)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
