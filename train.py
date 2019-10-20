import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam ,SGD, lr_scheduler
from torch.utils.data import DataLoader

import hawtorch
from hawtorch import io
from hawtorch import Trainer
from hawtorch.utils import backup
from hawtorch.metrics import ClassificationAverager

from models import *
from provider import load_data

args = io.load_json("configs_cora.json")

device = args["device"]
logger = io.logger(args["workspace_path"])

backup(args["workspace_path"])

def smooth_loss(x, As):
    N, f = x.shape
    I = torch.eye(N).to(x.device)
    loss = 0
    for A in As:
        L = I - A.squeeze()
        prior = x.permute(1,0) @ L @ x # [f, f]
        loss += prior.trace()
    return loss

def mix_loss(preds, truths, x, As):
    # cross entropy
    loss = F.cross_entropy(preds, truths)
    # GLR
    loss += args["lambda"] * smooth_loss(x, As)
    return loss


def create_loaders():
    # load data
    A, features, labels, idx_train, idx_val, idx_test = load_data(args["dataset"], args["keep_adj"])

    labels = np.argmax(labels, axis=1)

    args["fin"] = features.shape[1]
    args["num_classes"] = labels.max() + 1

    # to tensor
    A = torch.from_numpy(A.astype(np.float32)).to(device)
    features = torch.from_numpy(features.astype(np.float32)).to(device) # [M, Fin]
    labels = torch.LongTensor(labels).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)

    if args["dataset"] == "pubmed":
        #idx_other = torch.LongTensor(np.random.choice(np.arange(560,18717), 10000, replace=False)).to(A.device)
        idx_other = torch.LongTensor(np.arange(560,560+10000)).to(A.device)
        mask = torch.cat([idx_train, idx_val, idx_test, idx_other])
        idx_train = torch.LongTensor(torch.arange(0,60))
        idx_val = torch.LongTensor(torch.arange(60,560))
        idx_test = torch.LongTensor(torch.arange(560,1560))
        A = A[mask][:,mask]
        features = features[mask]
        labels = labels[mask]

    if args["keep_node"] < 1:
        num = int(args["keep_node"] * len(idx_train))
        idx_train = idx_train[:num]

    return {"A":A,
            "features":features,
            "labels":labels,
            "train_idx":idx_train,
            "test_idx":idx_test,
            "val_idx":idx_val,
            }



def create_trainer():
    logger.logblock(args)
    logger.info("Start creating trainer...")

    loaders = create_loaders()

    model = globals()[args["model"]](args, A=loaders["A"])

    objective = mix_loss

    optimizer = globals()[args["optimizer"]](model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args["lr_step"], gamma=args["lr_decay"])

    metrics = [ClassificationAverager(args["num_classes"]), ]

    trainer = Trainer(model, optimizer, scheduler, objective, device, loaders, logger,
                            metrics=metrics, 
                            workspace_path=args["workspace_path"],
                            eval_set="test",
                            report_step_interval=-1,
                            use_checkpoint="scratch",
                            )

    logger.logblock(model)
    logger.info("Trainer Created!")

    return trainer


if __name__ == "__main__":
    res = []
    for seed in range(10):
        logger.mute = True
        hawtorch.fix_random_seed(seed)
        trainer = create_trainer()
        trainer.train(args["epochs"])
        trainer.evaluate("test")
        res.append(trainer.stats["BestResult"])
        logger.mute = False
        logger.info(f"seed = {seed}: {trainer.stats['BestResult']}")
    logger.info(f"Statistics: {np.mean(res):.4f}, max: {np.max(res):.4f}, min: {np.min(res):.4f}")