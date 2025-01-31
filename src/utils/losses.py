# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

"""
Project: CL-Detection2024 Challenge Baseline
============================================

Loss fuction
损失函数

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_l1_loss(predictions, heatmaps):
    """
    Compute weighted L1 loss.
    
    Args:
        predictions (torch.Tensor): Predicted heatmaps from the model, shape (batch_size, channels, height, width)
        targets (torch.Tensor): Ground truth heatmaps, shape (batch_size, channels, height, width)
        heatmaps (torch.Tensor): Heatmaps to use for weighting, shape (batch_size, channels, height, width)
        
    Returns:
        torch.Tensor: Weighted L1 loss value
    """
    # Calculate the absolute difference between predictions and targets
    l1_loss = torch.abs(predictions - heatmaps)
    
    # Calculate the weight mask from heatmaps: add 1 to give a base weight of 1 to the background
    weight_mask = 25 * heatmaps + 1

    # Apply the weight mask to the L1 loss
    weighted_l1 = weight_mask * l1_loss
    
    # Compute the mean of the weighted L1 loss
    loss = torch.mean(weighted_l1)
    
    return loss

def focal_loss(predict, target):
    """
    This function computes the focal loss between the predicted heatmap and the target heatmap.
    该函数计算预测热图和目标热图之间的焦点损失。
    Args:
        predict (Tensor): Predicted heatmap tensor.
                          预测的热图张量。
        target (Tensor): Ground truth heatmap tensor.
                         真实热图张量。
    Returns:
        Tensor: Computed focal loss.
                计算出的焦点损失。
    """
    # clip predict heatmap range to prevent loss from becoming nan
    # 由于log2操作，限制热图预测值范围，防止loss NAN
    predict = torch.clamp(predict, min=1e-4, max=1 - 1e-4)

    pos_inds = target.gt(0.9)
    neg_inds = target.lt(0.9)
    neg_weights = torch.pow(1 - target[neg_inds], 4)  # negative weights | 负样本权重

    pos_pred = predict[pos_inds]
    neg_pred = predict[neg_inds]

    pos_loss = torch.log2(pos_pred) * torch.pow(
        1 - pos_pred, 2
    )  # positive loss | 正样本损失
    neg_loss = (
        torch.log2(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights
    )  # negative loss | 负样本损失

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = (
            -neg_loss
        )  # if no positive samples, only consider negative loss | 如果没有正样本，只考虑负样本损失
    else:
        loss = -(pos_loss + neg_loss) / num_pos  # average loss | 平均损失
    return loss

def dice_for_soft_targets_test2_2(x, y, apply_nonlin=None, eps=1e-6):
    """
    :param x: Network output, must be torch tensor of (b, c, x, y, z).
    :param y: Label, must be torch tensor of same shape as x.
    :return: Soft Dice loss.
    """
    assert all([i == j for i, j in zip(x.shape, y.shape)])

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    axes = list(range(2, len(x.shape)))

    tp = (F.relu(y - torch.abs(x - y), inplace=True)).sum(axes)
    fp = (F.relu(x - y, inplace=True)).sum(axes)
    fn = (F.relu(y - x, inplace=True)).sum(axes)

    return (- (2 * tp + eps) / (2 * tp + fp + fn + eps)).mean()


def load_loss(loss_name):
    """
    This function loads the appropriate loss function based on the provided loss name.
    该函数根据提供的损失名称加载适当的损失函数。
    Args:
        loss_name (str): The name of the loss function to load. Valid options are 'L1' and 'focalLoss'.
                         损失函数的名称。有效选项是'L1'和'focalLoss'。
    Returns:
        function: The corresponding loss function.
                  相应的损失函数。
    """
    if loss_name == "L1":
        loss_fn = nn.L1Loss()
    elif loss_name == "focalLoss":
        loss_fn = focal_loss
    elif loss_name == "dice":
        loss_fn = dice_for_soft_targets_test2_2
    elif loss_name == "weightedL1":
        loss_fn = weighted_l1_loss
    else:
        raise ValueError(
            "Please input valid model name, {} not in loss zone.".format(loss_name)
        )
    return loss_fn
