"""convert VOC format
+ region_voc
    + JPEGImages
    + SegmentationClass
"""

import os
import cv2
import h5py
import shutil
import argparse
import numpy as np
import os.path as osp
import concurrent.futures
from tqdm import tqdm
user_dir = osp.expanduser('~')


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--dataset', type=str, default='VisDrone',
                        choices=['VisDrone'], help='dataset name')
    parser.add_argument('--mode', type=str, default=['train','val'],
                        nargs='+', help='for train or val')
    parser.add_argument('--db_root', type=str,
                        default=user_dir+"/data/UnderWater",
                        # default="E:\\CV\\data\\Underwater\\UnderWater_VOC",
                        help="dataset's root path")
    parser.add_argument('--mask_size', type=list, default=[30, 40],
                        help="Size of production target mask")
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and region mask")
    args = parser.parse_args()
    return args


def show_image(img, labels, mask):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1).imshow(img[..., ::-1])
    plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    plt.subplot(2, 1, 2).imshow(mask)
    # plt.savefig('test_0.jpg')
    plt.show()


# copy train and test images
def _copy(src_image, dest_path):
    shutil.copy(src_image, dest_path)


def _resize(src_image, dest_path):
    img = cv2.imread(src_image)

    height, width = img.shape[:2]
    size = (int(width), int(height))

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    name = osp.basename(src_image)
    cv2.imwrite(osp.join(dest_path, name), img)


def _myaround_up(value):
    """0.05 * stride = 0.8"""
    tmp = np.floor(value).astype(np.int32)
    return tmp + 1 if value - tmp > 0.05 else tmp


def _myaround_down(value):
    """0.05 * stride = 0.8"""
    tmp = np.ceil(value).astype(np.int32)
    return max(0, tmp - 1 if tmp - value > 0.05 else tmp)


def _generate_mask(sample, mask_scale=(30, 40)):
    try:
        height, width = sample["height"], sample["width"]

        # Chip mask 40 * 30, model input size 640x480
        mask_h, mask_w = mask_scale
        region_mask = np.zeros((mask_h, mask_w), dtype=np.float32)

        for box in sample["bboxes"]:
            xmin = _myaround_down(1.0 * box[0] / width * mask_w)
            ymin = _myaround_down(1.0 * box[1] / height * mask_h)
            xmax = _myaround_up(1.0 * box[2] / width * mask_w)
            ymax = _myaround_up(1.0 * box[3] / height * mask_h)
            if xmin == xmax or ymin == ymax:
                continue
            region_mask[ymin:ymax+1, xmin:xmax+1] = 1

        return region_mask

    except Exception as e:
        print(e)
        print(sample["image"])


def load_anno(anno_xml):
    import xml.etree.ElementTree as ET
    CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')
    classes2id = {'holothurian': 0, 'echinus': 1, 'scallop': 2, 'starfish': 3, 'waterweeds': 4}
    box_all = []
    xml = ET.parse(anno_xml).getroot()
    pts = ['xmin', 'ymin', 'xmax', 'ymax']
    # bounding boxes
    for obj in xml.iter('object'):
        cls = obj.find('name').text
        if cls not in CLASSES:
            continue
        bbox = obj.find('bndbox')
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        bndbox.append(classes2id[cls])
        box_all += [bndbox]

    return np.array(box_all).astype(np.float32)


if __name__ == "__main__":
    mask_dir = "/home/twsf/data/UnderWater/val/region_mask"
    img_dir = "/home/twsf/data/UnderWater/val/images"
    anno_dir = "/home/twsf/data/UnderWater/val/labels"

    for filename in os.listdir(img_dir):
        img_shape = cv2.imread(osp.join(img_dir, filename)).shape[:2]
        sample = {
            'width': img_shape[1],
            'height': img_shape[0],
            'bboxes': load_anno(osp.join(anno_dir, filename[:-4]+'.xml')),
            'image': osp.join(img_dir, filename)
        }
        region_mask = _generate_mask(sample, [30, 40])
        basename = osp.basename(filename)
        maskname = osp.join(mask_dir, osp.splitext(basename)[0]+'.hdf5')
        with h5py.File(maskname, 'w') as hf:
            hf['label'] = region_mask
