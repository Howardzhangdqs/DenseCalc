import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Loss, self).__init__()
        self.device = device

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target[:, :2].float().to(pred.device) + torch.rand((target.size(0), 2), device=self.device) * 1e-3
        # print("Pred shape:", pred.shape)
        # print("Target shape:", target.shape)
        # print("Pred values:", pred)
        # print("Target values:", target)
        loss = F.binary_cross_entropy(F.sigmoid(pred), F.sigmoid(target))
        return loss
