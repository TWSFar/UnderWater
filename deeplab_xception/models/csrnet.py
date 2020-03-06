import torch.nn as nn
import torch
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M',
                              256, 256, 256, 'M', 512, 512, 512, 'M']
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = self.make_layers(self.frontend_feat, batch_norm=True)
        self.backend = self.make_layers(self.backend_feat,
                                        in_channels=512,
                                        batch_norm=True,
                                        dilation=True)
        self.output_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            model_dict = self.frontend.state_dict()
            new_dict = {}
            for k1, v1 in mod.state_dict().items():
                for k2, v2 in model_dict.items():
                    if k2 in k1 and v1.shape == v2.shape:
                        new_dict[k2] = v1
            model_dict.update(new_dict)
            self.frontend.load_state_dict(model_dict)

    def get_1x_lr_params(self):
        modules = [self.frontend]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.backend, self.output_layer]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels,
                                   v,
                                   kernel_size=3,
                                   padding=d_rate,
                                   dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


if __name__ == '__main__':
    model = CSRNet().cuda()
    input = torch.rand(2, 3, 255, 255).cuda()
    output = model(input)
    pass
