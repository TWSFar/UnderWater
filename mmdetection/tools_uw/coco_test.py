import os
import mmcv
import json
import argparse
import numpy as np
import os.path as osp
from mmdet.apis import init_detector, inference_detector, show_result
from pycocotools.coco import COCO
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--checkpoint', default="/home/twsf/work/underwater/mmdetection/tools_uw/work_dirs/map47_5/epoch_30.pth", help='model')
    parser.add_argument('--config', default='/home/twsf/work/underwater/mmdetection/tools_uw/config_new.py')
    parser.add_argument('--annotations', default='/home/twsf/data/UnderWater/val/instances_val.json')
    parser.add_argument('--test-dir', default='/home/twsf/data/UnderWater/val/images')
    parser.add_argument('--result-path', default='/home/twsf/data/UnderWater/val')
    args = parser.parse_args()
    return args


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == "__main__":
    args = parse_args()
    if not osp.exists(args.result_path):
        os.makedirs(args.result_path)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    coco_true = COCO(args.annotations)
    img_ids = coco_true.getImgIds()
    results = []

    for img_id in tqdm(img_ids):
        image_info = coco_true.loadImgs(img_id)[0]
        img_path = osp.join(args.test_dir, image_info['file_name'])
        result = inference_detector(model, img_path)
        # show_result(img_path, result, model.CLASSES)
        for i, boxes in enumerate(result):
            for box in boxes:
                box[2:4] = box[2:4] - box[:2]
                results.append({"image_id": img_id,
                                "category_id": i,
                                "bbox": np.round(box[:4]),
                                "score": box[4]})

    with open(os.path.join(args.result_path, 'coco_results.json'), "w") as f:
        json.dump(results, f, cls=MyEncoder)
        print("results json saved.")
