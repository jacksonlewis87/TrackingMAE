import torch

from enum import Enum


class Loss(Enum):
    L2 = "l2"


class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, labels):
        return torch.mean(torch.sum(torch.pow(labels - output, 2), dim=1))


def get_loss(loss: str):
    if loss == Loss.L2.value:
        return L2Loss()
    else:
        return None
