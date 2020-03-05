import os
import mmcv
import json
import argparse
import numpy as np
import os.path as osp
from mmdet.apis import init_detector, inference_detector, show_result
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('checkpoints', help='model')
    parser.add_argument('--config',
        default='/home/twsf/data/UnderWater/mmdetection/tools_uw/c_rcnn_101_32_chip.py')
    parser.add_argument('--test-dir', default='/home/twsf/data/UnderWater/test-A-image/')
    parser.add_argument('--result-path', default='/home/twsf/data/UnderWater/')
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


def main():
    args = parse_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config_file, args.checkpoint_file, device='cuda:0')

    img_list = os.listdir(args.test_dir)
    results = []

    f.writelines("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    for img_name in tqdm(img_list):
        img_path = osp.join(test_dir, img_name)
        result = inference_detector(model, img_path)
        for i, boxes in enumerate(result):
            for box in boxes:
                results.append({"image_id": img_name,
                                "category_id": i,
                                "bbox": np.round(box[:4]),
                                "score": box[4]})
        # show_result(img_path, result, model.CLASSES)

    with open(os.path.join(args.reults_path, 'results.json'), "w") as f:
        json.dump(results, f, cls=MyEncoder)
        print("results json saved.")
