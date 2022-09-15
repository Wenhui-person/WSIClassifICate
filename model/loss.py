
import os
import sys

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


class wsiLoss(nn.Module):
    def __init__(self, alph, gamma, beta):
        super(wsiLoss, self).__init__()
        self.alph = alph
        self.gamma = gamma
        self.beta = beta

    """
    def _kl_divergence_with_logits(self, p_logits, q_logits):
        p = torch.sigmoid(p_logits)
        log_p = F.logsigmoid(p_logits)
        log_q = F.logsigmoid(q_logits)
        kl = torch.sum(p * (log_p - log_q), -1)
        return kl
    

    def _mse_loss(self, p, q):
        p_sig = torch.sigmoid(p)
        q_sig = torch.sigmoid(q)

        return torch.mean(torch.pow(p_sig-q_sig, 2))
    """

    def fuck_loss(self, p):
        mask = torch.full_like(p, 0.5)

        return F.mse_loss(p, mask)

    def forward(self, y_pred, y_truth, sup_batch_size,
                unsup_batch_size, tsa_threshold):
        """
        y_pred:
            labeled_img
            unlabeled_img_ori
            unlabeled_img_linked
            unlabeled_img_aug
        :return:
        """
        y_pred = y_pred.view(-1)
        # print(y_pred)
        # print(y_truth)
        sup_pred = y_pred[:sup_batch_size]
        unsup_pred_ori = y_pred[sup_batch_size: sup_batch_size+unsup_batch_size]
        unsup_pred_linked = y_pred[sup_batch_size+unsup_batch_size: sup_batch_size+2*unsup_batch_size]
        unsup_pred_aug = y_pred[sup_batch_size+2*unsup_batch_size: ]

        # computer sup loss, add tsa
        sup_loss = F.binary_cross_entropy_with_logits(sup_pred, y_truth, reduction='none')
        # print("sup_loss", sup_loss)
        larger_than_threshold = torch.exp(-sup_loss) > tsa_threshold
        # print(larger_than_threshold)
        loss_mask = torch.ones_like(y_truth, dtype=torch.float32) * (1.0 - larger_than_threshold.type(torch.float32))
        # print(torch.sum(sup_loss * loss_mask, dim=-1))
        # print(torch.sum(loss_mask, dim=-1))
        # print(sup_loss.max())
        if loss_mask.sum() == 0.0:
            loss_1 = sup_loss.max()
        else:
            loss_1 = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1)).cuda()
        # loss_1 = F.binary_cross_entropy_with_logits(sup_pred, y_truth)

        # compute unsup loss
        unsup_pred_ori_sig = torch.sigmoid(unsup_pred_ori)
        unsup_pred_linked_sig = torch.sigmoid(unsup_pred_linked)
        unsup_pred_aug_sig = torch.sigmoid(unsup_pred_aug)

        loss_2 = F.mse_loss(unsup_pred_ori_sig, unsup_pred_aug_sig)
        # loss_3 = F.mse_loss(unsup_pred_ori_sig, unsup_pred_linked_sig)
        # loss_4 = F.mse_loss(unsup_pred_linked_sig, unsup_pred_aug_sig)

        # loss_5 = self.fuck_loss(unsup_pred_ori_sig) + self.fuck_loss(unsup_pred_linked_sig)\
        #          + self.fuck_loss(unsup_pred_aug_sig)


        loss = loss_1 + self.alph*loss_2 # + self.gamma*(loss_3+loss_4)#  - self.beta*(loss_5 - 1)

        return loss, loss_1, self.alph*loss_2,#  self.gamma*(loss_3+loss_4),# self.beta*(1-loss_5.item())