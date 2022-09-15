# -*- Coding: utf-8 -*-
import os
import argparse
import logging
import json
import time

import torch
from torch.optim import SGD
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model.loss import wsiLoss
from data.data_load import supImgDataset, unsupImgDataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale).item()
    else:
        threshold = None
        logging.info("Please chose your schedule.")

    output = threshold * (end - start) + start
    return output.cuda()


def train_epoch(summary, summary_writer, cfg, model,
                criterion, optimizer,
                dataloader_sup, dataloader_unsup):
    model.train()

    steps = len(dataloader_sup)
    dataiter_sup = iter(dataloader_sup)
    dataiter_unsup = iter(dataloader_unsup)

    total = len(dataloader_unsup) // len(dataloader_sup)
    # print(total)
    for i in range(total):
        for step in range(steps):
            data_sup, label_sup = next(dataiter_sup)
            # print(data_sup)
            # print(label_sup)
            data_ori, data_linked, data_aug = next(dataiter_unsup)
            data_sup = data_sup.cuda()
            label_sup = label_sup.float().cuda()
            data_ori, data_linked, data_aug = data_ori.cuda(), data_linked.cuda(), data_aug.cuda()
            sup_sum = data_sup.shape[0]
            unsup_sum = data_ori.shape[0]

            data = torch.cat([data_sup, data_ori, data_linked, data_aug]).float()
            # print(data.shape)
            output = model(data)
            # print(output[:sup_batch_size].sigmoid())
            # print(label_sup)
            if cfg["tsa"]:
                tsa_thresh = get_tsa_thresh(cfg["tsa"], summary['step'],
                                            total * steps * cfg["epoch"], 0.5, 1.0)
            else:
                tsa_thresh = 1.0
            # print("tsa thresh {}".format(tsa_thresh))
            """
            loss, sup_loss, aug_loss, link_loss, gap_loss = criterion(output,
                                                                      label_sup,
                                                                      sup_sum,
                                                                      unsup_sum,
                                                                      tsa_thresh)
            """
            loss, sup_loss, aug_loss = criterion(output, label_sup, sup_sum, unsup_sum, tsa_thresh)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = output[:sup_sum].sigmoid().view(-1)

            predicts = (probs >= 0.5).float().cuda()
            # print(predicts)
            # print((predicts == label_sup).sum().item())
            acc_data = (predicts == label_sup).sum().float().item() / sup_sum
            loss_data = loss.item()

            logging.info('{}, Epoch: {}, Step: {}, Training Loss: {:.5f}, Training Acc: {:.3f}\n'
                         'Sup Loss: {:.5f}, Unsup Loss: {:.5f}'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                                 summary['step'] + 1, loss_data, acc_data, sup_loss,
                                 aug_loss))

            summary['step'] += 1

            if summary['step'] % cfg['log_every'] == 0:
                summary_writer.add_scalar('train/loss', loss_data, summary['step'])
                summary_writer.add_scalar('train/acc', acc_data, summary['step'])
        dataiter_sup = iter(dataloader_sup)

    summary['epoch'] += 1

    return summary


def valid_epoch(summary, cfg, model, criterion,
                dataloader):
    batch_size = dataloader.batch_size
    steps = len(dataloader)
    dataiter = iter(dataloader)

    loss_sum = 0
    acc_sum = 0
    with torch.no_grad():
        for step in range(steps):
            X, y = next(dataiter)
            X = X.float().cuda()
            y = y.float().cuda()

            y_pred = model(X).view(-1)
            loss = F.binary_cross_entropy_with_logits(y_pred, y)

            probs = y_pred.sigmoid()
            predicts = (probs >= 0.5).float().cuda()
            # print(y)
            # print(predicts)
            acc_data = (predicts == y).sum().float().item() / batch_size
            # print(acc_data)
            loss_data = loss.item()
            loss_sum += loss_data
            acc_sum += acc_data

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary


def main(args):
    with open(args.config_path, "r") as f:
        cfg = json.load(f)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    with open(os.path.join(args.output_path, 'cfg.json'), "w") as f:
        json.dump(cfg, f, indent=1)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1)
    )
    model = model.cuda()
    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    # optimizer = Ranger(model.parameters(), lr=cfg['lr'])
    criterion = wsiLoss(alph=cfg["alph"], gamma=cfg["gamma"],
                        beta=cfg["beta"])

    # load data
    sup_dataset = supImgDataset(args.labeled_data_path, img_size=256,
                                crop_size=224)
    train_num = int(0.8 * len(sup_dataset))
    sup_dataset_train, sup_dataset_val = random_split(sup_dataset,
                                                      [train_num, len(sup_dataset) - train_num])
    unsup_dataset = unsupImgDataset(args.unlabeled_data_path, img_size=256,
                                    crop_size=224)
    dataloader_sup_train = DataLoader(sup_dataset_train, batch_size=cfg["sup_batch_size"],
                                      shuffle=True)
    dataloader_sup_val = DataLoader(sup_dataset_val, batch_size=cfg["sup_batch_size"] * 2)
    dataloader_unsup = DataLoader(unsup_dataset, batch_size=cfg["unsup_batch_size"],
                                  shuffle=True)

    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(os.path.join(args.output_path, "run"))
    loss_valid_best = float('inf')
    for epoch in range(cfg["epoch"]):
        summary_train = train_epoch(summary_train, summary_writer, cfg,
                                    model, criterion, optimizer,
                                    dataloader_sup_train, dataloader_unsup)
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'stat_dict': model.state_dict()},
                   os.path.join(args.output_path, "train.ckpt"))

        summary_valid = valid_epoch(summary_valid, cfg,
                                    model, criterion,
                                    dataloader_sup_val)

        logging.info('{}, Epoch : {}, Step : {}, Validation Loss : {:.5f}, Validation Acc : {:.3f}'
        .format(
            time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
            summary_train['step'], summary_valid['loss'],
            summary_valid['acc']))

        summary_writer.add_scalar(
            'valid/step', summary_valid['loss'], summary_train['step'])
        summary_writer.add_scalar(
            'valid/acc', summary_valid['acc'], summary_train['step'])

        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']
            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.state_dict()},
                       os.path.join(args.output_path, "best.ckpt"))

        summary_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MIL model")
    parser.add_argument('--labeled_data_path',
                        default="/home/qianslab/yangwenhui/TUMOR_data_processed/bag_files/data_enrichment/Labeled_Trainset_path_with_16800_samples.npy",
                        type=str,
                        help="The path to the labeled dataset.")
    parser.add_argument('--unlabeled_data_path',
                        default="/home/qianslab/yangwenhui/TUMOR_data_processed/bag_files/data_enrichment/Unlabeled_Trainset_path_with_780000_.npy",
                        type=str,
                        help="The path to the unlabeled dataset.")
    parser.add_argument('--config_path',
                        default="/home/qianslab/yangwenhui/Multiple Instance Learning manuscript1/code/wsiClassification/configs/train_cfg.json",
                        type=str,
                        help="The path to the training config file.")
    parser.add_argument('--output_path', default="/home/qianslab/yangwenhui/results", type=str,
                        help="The path to save result and model.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args)

    # --labeled_data_path /mnt/data/Cervical-cancer-data/wwang/train_sup_patch/Labeled_Trainset_path_with_6150_samples.npy
    # --unlabeled_data_path /mnt/data/Cervical-cancer-data/wwang/train_sup_patch/Unlabeled_Trainset_path_with_492000_.npy
    # --config_path configs/train_cfg.json
    # --output_path /mnt/data/Cervical-cancer-data/wwang/RESULTS
