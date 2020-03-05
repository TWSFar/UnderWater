import os
import glob
import pickle
import numpy as np
import os.path as osp
from PIL import Image
import xml.etree.ElementTree as ET
IMG_ROOT = "JPEGImages"
ANNO_ROOT = "Annotations"


class UnderWater(object):
    def __init__(self, db_root):
        self.db_root = db_root
        self.image_set = db_root + '/ImageSets/Main'
        self.region_voc_dir = db_root + '/region_mask'
        self.detect_voc_dir = db_root + '/region_chip'
        self.cache_dir = osp.join(db_root, 'cache')
        self._init_path()

    def _init_path(self):
        if not osp.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_imglist(self, split='train'):
        """ return list of all image paths
        """
        with open(osp.join(self.image_set, split+'.txt'), 'r') as f:
            return [osp.join(self.db_root, IMG_ROOT, x.strip()+'.jpg') for x in f.readlines()]

    def _get_annolist(self, split):
        """ annotation type is '.txt'
        return list of all image annotation path
        """
        img_list = self._get_imglist(split)
        return [img.replace(IMG_ROOT, ANNO_ROOT).replace('jpg', 'txt')
                for img in img_list]

    def _get_gtbox(self, anno_path):
        box_all = []
        gt_cls = []
        xml = ET.parse(anno_path).getroot()
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        # bounding boxes
        for obj in xml.iter('object'):
            cls = obj.find('name').text
            if cls == "waterweeds":
                continue
            bbox = obj.find('bndbox')
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            box_all += [bndbox]
            gt_cls.append(cls)

        return {'bboxes': np.array(box_all, dtype=np.float64),
                'cls': gt_cls}

    def _load_samples(self, split):
        cache_file = osp.join(self.cache_dir, split + '_samples.pkl')

        # load bbox and save to cache
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                samples = pickle.load(fid)
            print('gt samples loaded from {}'.format(cache_file))
            return samples

        img_list = self._get_imglist(split)
        sizes = [Image.open(img).size for img in img_list]

        anno_path = [img_path.replace(IMG_ROOT, ANNO_ROOT).replace('jpg', 'xml')
                        for img_path in img_list]
        # load information of image and save to cache
        samples = [self._get_gtbox(ann) for ann in anno_path]

        for i, img_path in enumerate(img_list):
            samples[i]['image'] = img_path  # image path
            samples[i]['width'] = sizes[i][0]
            samples[i]['height'] = sizes[i][1]

        with open(cache_file, 'wb') as fid:
            pickle.dump(samples, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt samples to {}'.format(cache_file))

        return samples


if __name__ == "__main__":
    dataset = UnderWater("E:\\CV\\data\\Underwater\\UnderWater_VOC")
    out = dataset._load_samples('train')
    pass
