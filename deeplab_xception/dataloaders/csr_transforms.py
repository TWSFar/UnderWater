import cv2
import numbers
import random
import numpy as np
from PIL import Image, ImageOps
from skimage import transform as sktsf

import torch
import torchvision.transforms as stf
import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx


# ===============================img tranforms============================

class RandomHorizontallyFlip(object):
    def __call__(self, img, den):
        x_flip = random.choice([True, False])
        if x_flip:
            img = img[:, ::-1, :]
            den = den[:, ::-1]

        return img, den


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, dst_size=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        if dst_size is None:
            th, tw = self.size
        else:
            th, tw = dst_size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, den):
        img = cv2.resize(img, self.size)
        den = sktsf.resize(den, img.shape[:2])
        return img, den


# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor*self.para
        return tensor


class GTScaleDown(object):
    def __init__(self, input_size, factor=8):
        self.factor = factor
        self.new_h = int(input_size[1] / factor)
        self.new_w = int(input_size[0] / factor)
        self.gamma = factor ** 2

    def __call__(self, den):
        if self.factor == 1:
            return den
        den = sktsf.resize(den, (self.new_h, self.new_w)) * self.gamma

        return den


class transfrom(object):
    def __init__(self, train=True):
        self.train = train
        self.train_main_transform = Compose([
                FreeScale(opt.input_size),
                RandomHorizontallyFlip()
            ])
        self.val_main_transform = Compose([
                FreeScale(opt.input_size),
            ])
        self.img_transform = stf.Compose([
                stf.ToTensor(),
                stf.Normalize(opt.mean, opt.std)
            ])
        self.gt_transform = stf.Compose([
                GTScaleDown(opt.input_size, opt.gtdownrate),
                # log_para is a factor of density
                LabelNormalize(opt.log_para)
            ])

    def __call__(self, img, den):
        if self.train:
            img, den = self.train_main_transform(img, den)
        else:
            img, den = self.val_main_transform(img, den)
        img = self.img_transform(img.copy())
        den = self.gt_transform(den)

        return img, den


class re_transform(object):
    def __init__(self):
        self.restore_transform = stf.Compose([
                DeNormalize(opt.mean, opt.std),
                stf.ToPILImage()
            ])

    def __call__(self, img):
        return self.restore_transform(img)


re_tsf = re_transform()
