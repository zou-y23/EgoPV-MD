# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch.nn as nn
import torch

# Custom loss by Taein. 
class euclidean_loss:
    def __init__(self, reduction=None):
        self.reduction = reduction

    def __call__(self,output, target):
        #point_dist = torch.abs(torch.sum((output - target) ** 2,1))**0.5
        point_dist = torch.linalg.norm(output-target, dim=1)
        loss = torch.mean(point_dist)
        return loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "L1Loss": nn.L1Loss,
    "mse": nn.MSELoss,
    "l2": euclidean_loss,
}

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
