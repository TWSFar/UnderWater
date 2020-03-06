import torch
import torch.nn as nn


class MSELoss(object):
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, pred, target):
        device = pred.device
        criterion = nn.MSELoss(reduction=self.reduction).to(device)

        loss = criterion(pred, target.unsqueeze(1))

        return loss


if __name__ == "__main__":
    torch.manual_seed(1)
    loss = MSELoss()
    a = torch.rand(2)
    b = torch.tensor([0, 0])
    print(loss(a, b))
    pass
