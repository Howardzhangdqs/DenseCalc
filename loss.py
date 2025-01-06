import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target[:, :2].float() + torch.random((target.size(0), 2)) * 1e-3
        loss = F.binary_cross_entropy(pred, target)
        return loss
