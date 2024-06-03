import torch

from enum import Enum


class Loss(Enum):
    L2 = "l2"
    BCE = "bce"


class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, labels):
        return torch.mean(torch.sum(torch.pow(labels - output, 2), dim=1))


class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, output, labels):
        loss = labels * torch.log(output + 1e-7) + (1 - labels) * torch.log(1 - output + 1e-7)
        return -torch.mean(loss)


def get_loss(loss: str):
    if loss == Loss.L2.value:
        return L2Loss()
    elif loss == Loss.BCE.value:
        return BCELoss()
    else:
        return None
