import os
import cv2
import json
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import nms, plot_img, show_image
CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish' 'region')


def parse_args():
    parser = argparse.ArgumentParser(description='UnderWaterDataset submit')
    parser.add_argument('--split', type=str, default='test', help='split')
    parser.add_argument('--result_file', type=str,
                        default="/home/twsf/data/UnderWater/val/results.json")
    parser.add_argument('--loc_dir', type=str,
                        default='/home/twsf/data/UnderWater/val/region_loc/')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--img_dir', type=str, help="show image path",
                        default="/home/twsf/data/UnderWater/test/images")
    args = parser.parse_args()
    print('# parametes list:')
    for (key, value) in args.__dict__.items():
        print(key, '=', value)
    print('')
    return args


def Combine():
    args = parse_args()

    loc_file = osp.join(args.loc_dir, args.split + "_chip.json")

    with open(args.result_file, 'r') as f:
        results = json.load(f)
    with open(loc_file, 'r') as f:
        chip_loc = json.load(f)

    detecions = dict()
    for det in tqdm(results):
        img_id = det['image_id']
        cls_id = det['category_id']
        bbox = det['bbox']
        score = det['score']
        loc = chip_loc[img_id]
        bbox = [bbox[0] + loc[0], bbox[1] + loc[1], bbox[2] + loc[0], bbox[3] + loc[1]]
        img_name = '_'.join(img_id.split('_')[:-1]) + osp.splitext(img_id)[1]
        if img_name in detecions:
            detecions[img_name].append(bbox + [score, cls_id])
        else:
            detecions[img_name] = [bbox + [score, cls_id]]

    output_file = 'DET_results-%s' % args.split + '.csv'

    with open(output_file, 'w') as f:
        f.writelines("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
        for img_name, det in tqdm(detecions.items()):
            det = nms(det, score_threshold=0.5)
            img_id = osp.splitext(img_name)[0] + '.xml'
            for box in det:
                f.writelines(CLASSES[int(box[5])]+','+img_id+','+str(box[4]))
                for v in np.round(box[:4]):
                    f.writelines(','+str(int(v)))
                f.writelines('\n')

            if args.show:
                img_path = osp.join(args.img_dir, img_name)
                img = cv2.imread(img_path)[:, :, ::-1]
                bboxes = det[:, [0, 1, 2, 3, 5, 4]]
                # show_image(img, bboxes)
                img = plot_img(img, bboxes, CLASSES)
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 1, 1).imshow(img)
                plt.show()


if __name__ == '__main__':
    Combine()
