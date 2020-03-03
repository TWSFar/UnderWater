import os
import cv2
import json
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import OrderedDict

hyp = {
    'dataset': 'UnderWater',
    'img_type': '.jpg',
    'mode': 'train',  # for save instance_train.json
    'data_dir': '/home/twsf/data/UnderWater',
}
hyp['json_dir'] = osp.join(hyp['data_dir'], 'Annotations_json')
hyp['xml_dir'] = osp.join(hyp['data_dir'], 'Annotations')
hyp['img_dir'] = osp.join(hyp['data_dir'], 'JPEGImages')
hyp['set_file'] = osp.join(hyp['data_dir'], 'ImageSets', 'Main', hyp['mode'] + '.txt')

classes2id = {'holothurian': 0, 'echinus': 1, 'scallop': 2, 'starfish': 3, 'waterweeds': 4}


class getItem(object):
    def __init__(self):
        self.classes = ('holothurian', 'echinus', 'scallop', 'starfish')
        self.classes2id = classes2id
        # self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def get_img_item(self, file_name, image_id, size):
        """Gets a image item."""
        image = OrderedDict()
        image['file_name'] = file_name
        image['height'] = int(size['height'])
        image['width'] = int(size['width'])
        image['id'] = image_id
        return image

    def get_ann_item(self, bbox, img_id, cat_id, anno_id):
        """Gets an annotation item."""
        x1 = bbox[0]
        y1 = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        annotation = OrderedDict()
        annotation['segmentation'] = [[x1, y1, x1, (y1 + h), (x1 + w), (y1 + h), (x1 + w), y1]]
        annotation['area'] = w * h
        annotation['iscrowd'] = 0
        annotation['image_id'] = img_id
        annotation['bbox'] = [x1, y1, w, h]
        annotation['category_id'] = cat_id
        annotation['id'] = anno_id
        return annotation

    def get_cat_item(self):
        """Gets an category item."""
        categories = []
        for idx, cat in enumerate(self.classes):
            cate = {}
            cate['supercategory'] = cat
            cate['name'] = cat
            cate['id'] = idx
            categories.append(cate)

        return categories


def getGTBox(anno_xml, **kwargs):
    box_all = []
    gt_cls = []
    xml = ET.parse(anno_xml).getroot()
    pts = ['xmin', 'ymin', 'xmax', 'ymax']
    # bounding boxes
    for obj in xml.iter('object'):
        cls = classes2id[obj.find('name').text]
        if cls == 4:
            continue
        bbox = obj.find('bndbox')
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        box_all += [bndbox]
        gt_cls.append(int(cls))

    return box_all, gt_cls


def make_json():
    item = getItem()
    images = []
    annotations = []
    anno_id = 0

    # categories
    categories = item.get_cat_item()

    with open(hyp['set_file'], 'r') as f:
        xml_list = f.readlines()
    for id, file_name in enumerate(tqdm(xml_list)):
        img_id = id

        # anno info
        anno_xml = os.path.join(hyp['xml_dir'], file_name.strip() + '.xml')
        box_all, gt_cls = getGTBox(anno_xml)
        for ii in range(len(box_all)):
            annotations.append(
                item.get_ann_item(box_all[ii], img_id, gt_cls[ii], anno_id))
            anno_id += 1

        # image info
        xml = ET.parse(anno_xml).getroot()
        img_name = file_name.strip() + '.jpg'  # image name
        img_path = osp.join(hyp['img_dir'], img_name)  # image path
        assert osp.isfile(img_path)
        # tsize = xml.find('size')
        # size = {'height': int(tsize.find('height').text),
        #         'width': int(tsize.find('width').text)}
        tsize = cv2.imread(img_path).shape[:2]
        size = {'height': int(tsize[0]),
                'width': int(tsize[1])}
        image = item.get_img_item(img_name, img_id, size)
        images.append(image)

    # all info
    ann = OrderedDict()
    ann['images'] = images
    ann['categories'] = categories
    ann['annotations'] = annotations

    # saver
    if not osp.exists(hyp['json_dir']):
        os.makedirs(hyp['json_dir'])
    save_file = os.path.join(hyp['json_dir'], 'instances_{}.json'.format(hyp['mode']))
    print('Saving annotations to {}'.format(save_file))
    json.dump(ann, open(save_file, 'w'), indent=4)


if __name__ == '__main__':
    make_json()
