import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, distribution=[1, 1], device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Loss, self).__init__()
        self.device = device
        self.distribution = distribution

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target[:, :2].float().to(pred.device) + torch.rand((target.size(0), 2), device=self.device) * 1e-3

        loss = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=torch.tensor(self.distribution).to(self.device)
        )
        return loss
