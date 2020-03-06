import cv2
import random
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter
import torch


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        assert max(mean) <= 1 and max(std) <= 1, "mean or std value error!"
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] = sample['image'].astype(np.float32)
        if sample['label'] is not None:
            sample['label'] = sample['label'].astype(np.float32)
        sample['image'] /= 255.0
        sample['image'] -= self.mean
        sample['image'] /= self.std

        return sample


class RandomColorJeter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.tr = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        sample['image'] = self.tr(Image.fromarray(sample['image']))
        sample['image'] = np.array(sample['image'])

        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = Image.fromarray(sample['image'])
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            sample['image'] = np.array(sample['image'])

        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample['image'] = sample['image'][:, ::-1, :]
            sample['label'] = sample['label'][:, ::-1]

        return sample


class FixedNoMaskResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size  # size: (w, h)

    def __call__(self, sample):
        sample['image'] = cv2.resize(sample['image'], self.size)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample['image'] = torch.from_numpy(sample['image']).permute(2, 0, 1)
        if sample['label'] is not None:
            sample['label'] = torch.from_numpy(sample['label'])

        return sample


if __name__ == "__main__":
    from torchvision import transforms
    img = cv2.imread("/home/twsf/work/CRGNet/data/Visdrone_Region/JPEGImages/0000001_02999_d_0000005.jpg")
    gt = cv2.imread("/home/twsf/work/CRGNet/data/Visdrone_Region/SegmentationClass/0000001_02999_d_0000005.png")
    pair = {'image': img, 'label': gt}
    model = transforms.Compose([
            FixedNoMaskResize(size=(640, 480)),
            RandomColorJeter(0.3, 0.3, 0.3, 0.3),
            RandomHorizontalFlip(),
            Normalize(),
            ToTensor()])
    sample = model(pair)
    pass
