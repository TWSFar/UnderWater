from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

gt_file = "/home/twsf/data/UnderWater/val/instances_val.json"
pred_file = "/home/twsf/data/UnderWater/val/coco_results.json"


if __name__ == '__main__':
    coco_true = COCO(gt_file)
    coco_pred = coco_true.loadRes(pred_file)
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = coco_true.getImgIds()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
