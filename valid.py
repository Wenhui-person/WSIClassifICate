# -*- Coding: utf-8 -*-
import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import auc, accuracy_score, roc_curve

from data.data_load import testDataset


def valid(model, dataloader):
    steps = len(dataloader)
    dataiter = iter(dataloader)

    y_probs = []
    y_truth = []
    with torch.no_grad():
        for step in range(steps):
            X, y = next(dataiter)
            X = X.float().cuda()
            y_truth.append(y.float().item())

            y_pred = model(X).view(-1)
            probs = y_pred.sigmoid().cpu().item()
            y_probs.append(probs)
    y_probs = torch.from_numpy(np.array(y_probs)).view(-1)
    y_truth = torch.from_numpy(np.array(y_truth)).view(-1)
    return y_probs, y_truth


def test_metrics(y_prob, y_truth):
    fpr, tpr, thresholds = roc_curve(y_truth, y_prob, pos_label=1)
    auc_score = auc(fpr, tpr)
    return auc_score


def main(args):
    dataset = testDataset(args.data_path, 256, 224)
    dataloader = DataLoader(dataset, batch_size=196)

    model = resnet18()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1)
    )
    model_dict = torch.load(args.model_path)
    model.load_state_dict(model_dict["state_dict"])
    model.cuda()

    y_probs, y_truth = valid(model, dataloader)
    auc_score = test_metrics(y_probs, y_truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model.")
    parser.add_argument('--data_path', default="/home/qianslab/yangwenhui/TUMOR_data_processed/bag_files/data_enrichment/test/Labeled_Trainset_path_with_24000_samples.npy", type=str,
                        help="Tha path to testset.")
    parser.add_argument('--model_path', default="/home/qianslab/yangwenhui/results/best_1.ckpt", type=str,
                        help="The path of the model.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args)
