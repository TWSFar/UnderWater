"""
use rondom array replace objce witch was neglected
"""
import os
import cv2
import sys
import h5py
import json
import argparse
import numpy as np
import os.path as osp

import utils
user_dir = osp.expanduser('~')


def parse_args():
    parser = argparse.ArgumentParser(description="convert to chip dataset")
    parser.add_argument('--test_dir', type=str,
                        default=user_dir+"/data/UnderWater/test")
    parser.add_argument('--side', type=int, default=800, help="window base side")
    parser.add_argument('--stride', type=int, default=600, help="window stride")
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and chip box")
    args = parser.parse_args()
    return args


args = parse_args()
print(args)


class MakeDataset(object):
    def __init__(self):
        self.img_dir = osp.join(args.test_dir, "images")
        self.chip_dir = osp.join(args.test_dir, "region_chip")
        self.loc_dir = osp.join(args.test_dir, "region_loc")
        self._init_path()

    def _init_path(self):
        if not osp.exists(self.chip_dir):
            os.makedirs(self.chip_dir)
        if not osp.exists(self.loc_dir):
            os.makedirs(self.loc_dir)

    def __call__(self):
        print("make test detect dataset...")
        chip_ids = []
        chip_loc = dict()
        img_list = os.listdir(self.img_dir)
        assert len(img_list) > 0
        for i, img_name in enumerate(img_list):
            img_id = osp.splitext(osp.basename(img_name))[0]
            sys.stdout.write('\rcomplete: {:d}/{:d} {:s}'
                                .format(i + 1, len(img_list), img_id))
            sys.stdout.flush()

            chiplen, loc = self.make_chip(img_name)
            for i in range(chiplen):
                chip_ids.append('{}_{}'.format(img_id, i))
            chip_loc.update(loc)

        # wirte chip loc json
        with open(osp.join(self.loc_dir, 'test_chip.json'), 'w') as f:
            json.dump(chip_loc, f)
            print('write loc json')

    def make_chip(self, img_name):
        image = cv2.imread(osp.join(self.img_dir, img_name))
        height, width = image.shape[:2]
        img_id = osp.splitext(osp.basename(img_name))[0]

        # make chip
        if max(height, width) < 1200:
            side = min(height, width)
            stride = max(side - 100, 100)
        elif max(height, width) < 2200:
            side = max(height, width) * 0.5
            stride = max(side - 300, 300)
        else:
            side = max(height, width) * 0.4
            stride = max(side - 300, 500)
        region_box = self.shift(width, height, side, stride)

        if args.show:
            utils.show_image(image, region_box)

        chip_loc = self.write_chip_and_anno(image, img_id, region_box)

        return len(region_box), chip_loc

    def shift(self, width, height, side, stride):
        w_num = max(round((width - (side - stride)) / stride), 1)
        w_len = np.ceil(width / w_num) + (side - stride)
        h_num = max(round((height - (side - stride)) / stride), 1)
        h_len = np.ceil(height / h_num) + (side - stride)

        shift_w = np.arange(0, w_num) * stride
        shift_h = np.arange(0, h_num) * stride

        shift_w, shift_h = np.meshgrid(shift_w, shift_h)
        shifts = np.vstack((
            shift_w.ravel(), shift_h.ravel(),
            shift_w.ravel(), shift_h.ravel()
        )).transpose()
        shifts[:, 2] = (shifts[:, 2] + w_len).clip(0, width)
        shifts[:, 3] = (shifts[:, 3] + h_len).clip(0, height)

        return shifts.astype(np.int32)

    def write_chip_and_anno(self, image, img_id, chip_list):
        """write chips of one image to disk and make xml annotations
        """
        chip_loc = dict()
        for i, chip in enumerate(chip_list):
            img_name = '{}_{}.jpg'.format(img_id, i)
            chip_loc[img_name] = [int(x) for x in chip]
            chip_img = image[chip[1]:chip[3], chip[0]:chip[2], :].copy()
            assert len(chip_img.shape) == 3

            cv2.imwrite(osp.join(self.chip_dir, img_name), chip_img)

        return chip_loc


if __name__ == "__main__":
    makedataset = MakeDataset()
    makedataset()
