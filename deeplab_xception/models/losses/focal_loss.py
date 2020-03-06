import torch
import torch.nn as nn


class FocalLoss(object):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=255, weight=None):
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight

    def __call__(self, logit, target):
        device = logit.device

        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction='none').to(device)

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)

        one_hot = target > 0
        t = torch.ones_like(one_hot, dtype=torch.float)
        alpha = torch.where(one_hot, self.alpha * t, (1 - self.alpha) * t)
        logpt = logpt * alpha
        loss = -((1 - pt) ** self.gamma) * logpt

        return loss.mean()


if __name__ == "__main__":
    loss = FocalLoss()
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss(a, b))
