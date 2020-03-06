import torch
import torch.nn as nn
# import sys
# import os.path as osp
# sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from .necks import ASPP
from .backbones import build_backbone
from .sync_batchnorm import SynchronizedBatchNorm2d


class DeepLab(nn.Module):
    def __init__(self, opt):
        super(DeepLab, self).__init__()

        if opt.sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(opt.backbone, opt.output_stride, BatchNorm)
        self.aspp = ASPP(opt.backbone,
                         opt.output_stride,
                         self.backbone.high_outplanes+128,
                         BatchNorm)
        self.link_conv = nn.Sequential(nn.Conv2d(
            self.backbone.low_outplanes, 128, kernel_size=1, stride=1, padding=0, bias=False))
        self.last_conv = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Conv2d(128, opt.output_channels, kernel_size=1, stride=1))

        self._init_weight()
        if opt.freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        low_level_feat = self.link_conv(low_level_feat)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.aspp(x)
        x = self.last_conv(x)
        if x.shape[1] > 1:
            return x.sigmoid()
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.last_conv, self.link_conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def _init_weight(self):
        for module in [self.link_conv, self.last_conv]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, SynchronizedBatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenetv2', output_stride=16)
    model.eval()
    input = torch.rand(5, 3, 640, 480)
    output = model(input)
    pass
