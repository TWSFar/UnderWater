import os
import cv2
import h5py
import os.path as osp
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
from dataloaders import deeplab_transforms as dtf
IMG_ROOT = "JPEGImages"
REGION_ROOT = "SegmentationClass"


class UnderWaterRegion(Dataset):
    """
    underwater dataset
    """
    nclass = 2

    def __init__(self, opt, mode="train"):
        super().__init__()
        self.data_dir = opt.root_dir
        self.mode = mode

        self.img_dir = osp.join(self.data_dir, IMG_ROOT, '{}.jpg')
        self.label_dir = osp.join(self.data_dir, REGION_ROOT, '{}.hdf5')
        self.img_ids = self._load_image_set_index()

        self.img_number = len(self.img_ids)

        # transform
        if self.mode == "train":
            self.transform = transforms.Compose([
                dtf.FixedNoMaskResize(size=opt.input_size),
                dtf.RandomColorJeter(0.3, 0.3, 0.3, 0.3),
                dtf.RandomHorizontalFlip(),
                dtf.Normalize(opt.mean, opt.std),
                dtf.ToTensor()])
        else:
            self.transform = transforms.Compose([
                dtf.FixedNoMaskResize(size=opt.input_size),  # 513
                dtf.Normalize(opt.mean, opt.std),
                dtf.ToTensor()])

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_index = []
        image_set_file = self.data_dir \
            + "/ImageSets/{}.txt".format(self.mode)

        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file, 'r') as f:
            for line in f.readlines():
                image_index.append(line.strip())
        return image_index

    def __getitem__(self, index):
        id = self.img_ids[index]
        img_path = self.img_dir.format(id)
        label_path = self.label_dir.format(id)
        assert osp.isfile(img_path), '{} not exist'.format(img_path)
        assert osp.isfile(label_path), '{} not exist'.format(label_path)

        img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
        with h5py.File(label_path, 'r') as hf:
            label = np.array(hf['label'])

        o_h, o_w = img.shape[:2]

        sample = {"image": img, "label": label}
        sample = self.transform(sample)

        scale = torch.tensor([sample["image"].shape[1] / o_h,
                              sample["image"].shape[0] / o_w])
        sample["scale"] = scale
        sample["path"] = img_path

        return sample

    def __len__(self):
        return len(self.img_ids)


if __name__ == "__main__":
    # from torch.utils.data import DataLoader
    # from configs.deeplabv3_region_sample import opt
    dataset = UnderWaterRegion(opt, mode="train")
    data = dataset.__getitem__(0)
    dataloader = DataLoader(dataset, batch_size=20, num_workers=2, shuffle=True)
    for sample in dataloader:
        pass
    pass
