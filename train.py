import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import numpy as np
import wandb
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
)

from sklearn.metrics import roc_auc_score

from dataset import CoswaraDataset
from models import M5 
from utils import same_seeds

N_SOUND = 9

def train(accelerator, args, data_loader, model, optimizer, criterion, scheduler=None):
    train_loss = []
    train_accs = []
    log_loss = []
    log_accs = []
    all_labels, all_logits = [], []

    model.train()

    for idx, batch in enumerate(tqdm(data_loader)):
        logits = model(batch['wav'])
        labels = batch['label']
        loss = criterion(logits, torch.eye(2)[labels].to(accelerator.device))
        acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
        loss = loss / args.accu_step
        accelerator.backward(loss)

        all_labels.extend(labels.cpu().numpy())
        if not len(all_logits):
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate([all_logits, logits.detach().cpu().numpy()])
        train_loss.append(loss.item())
        train_accs.append(acc)
        log_loss.append(loss.item())
        log_accs.append(acc)

        if ((idx + 1) % args.accu_step == 0) or (idx == len(data_loader) - 1):
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        if ((idx + 1) % args.log_step == 0) or (idx == len(data_loader) - 1):
            log_loss = sum(log_loss) / len(log_loss)
            log_acc = sum(log_accs) / len(log_accs)
            print(f"Train Loss: {log_loss:.4f}, Train Acc: {log_acc:.4f}")
            log_loss, log_acc= [], []
    
    train_auc = roc_auc_score(np.eye(2)[all_labels], all_logits)
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    return train_loss, train_acc, train_auc


@torch.no_grad()
def validate(data_loader, model, criterion):
    model.eval()
    valid_loss = []
    valid_accs = []
    all_labels, all_logits = [], []

    for idx, batch in enumerate(tqdm(data_loader)):
        logits = model(batch['wav'])
        labels = batch['label']
        loss = criterion(logits, torch.eye(2)[labels].to(accelerator.device))
        acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()

        all_labels.extend(labels.cpu().numpy())
        if not len(all_logits):
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate([all_logits, logits.detach().cpu().numpy()])
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_auc = roc_auc_score(np.eye(2)[all_labels], all_logits) 
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    return valid_loss, valid_acc, valid_auc


def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator()

    model = M5(1,2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    criterion = nn.BCELoss()

    starting_epoch = 1
    if args.wandb:
        wandb.watch(model)

    dataset = CoswaraDataset(
        csv_path = os.path.join(args.data_dir, "combined_data.csv"),
        audio_path = args.data_dir,
        label_mapping = os.path.join(args.data_dir, "mapping.json"),
    )

    train_size = int(round(len(dataset) * 0.8))
    valid_size = len(dataset) - train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

    print("Size of Training Set:", train_size * args.batch_size * N_SOUND)
    print("Size of Validation Set:", valid_size * args.batch_size * N_SOUND)

    train_loader = DataLoader(
        train_set,
        collate_fn=train_set.dataset.collate_fn,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set,
        collate_fn=valid_set.dataset.collate_fn,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True
    )
    warmup_step = int(0.1 * len(train_loader)) // args.accu_step
    scheduler = None
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer, warmup_step, args.num_epoch * len(train_loader) - warmup_step
    # )

    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )
    best_loss = float("inf")

    for epoch in range(starting_epoch, args.num_epoch + 1):
        print(f"Epoch {epoch}:")
        train_loss, train_acc, train_auc = train(
            accelerator, args, train_loader, model, optimizer, criterion, scheduler
        )
        valid_loss, valid_acc, valid_auc = validate(valid_loader, model, criterion)
        print(f"Train Accuracy: {train_acc:.2f}, Train Loss: {train_loss:.2f}, Train AUC: {train_auc:.2f}")
        print(f"Valid Accuracy: {valid_acc:.2f}, Valid Loss: {valid_loss:.2f}, Valid AUC: {valid_auc:.2f}")
        if args.wandb:
            wandb.log(
                {
                    "Train Accuracy": train_acc,
                    "Train Loss": train_loss,
                    "Validation Accuracy": valid_acc,
                    "Validation Loss": valid_loss,
                }
            )
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                {
                    "name": args.model_name,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.ckpt_dir, f"{args.prefix}_best.ckpt"),
            )
        torch.save(
            {
                "name": args.model_name,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.ckpt_dir, f"{args.prefix}_latest.ckpt"),
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=5920)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./coswara",
    )
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=96000)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--wd", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

    # training
    parser.add_argument("--num_epoch", type=int, default=3)
    parser.add_argument("--accu_step", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_step", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    if args.wandb:
        wandb.login()
        wandb.init(project="AI")
        wandb.config.update(args)
    main(args)
