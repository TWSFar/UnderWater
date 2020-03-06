import torch
import torch.nn as nn


class CrossEntropyLoss(object):
    def __init__(self, weight=None, ignore_index=-1):
        self.ignore_index = ignore_index
        self.weight = torch.tensor(weight).float()

    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        device = logit.device
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction='mean').to(device)

        loss = criterion(logit, target.long())

        return loss


if __name__ == "__main__":
    torch.manual_seed(1)
    loss = CrossEntropyLoss()
    a = torch.rand(1, 2, 1, 1).cuda()
    b = torch.tensor([[[1.]]]).cuda()
    print(loss(a, b))
    pass
