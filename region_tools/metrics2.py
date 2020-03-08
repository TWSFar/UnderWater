"""
直接使用测试集的原图进行验证
"""
import os
import cv2
import json
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from utils import nms
ANNOTATION_DIR = "/home/twsf/data/UnderWater/val/labels/"
CHIP_RESULT_FILE = "/home/twsf/data/UnderWater/val/results2.json"
LOCATION_FILE = "/home/twsf/data/UnderWater/val/region_loc/test_chip.json"
CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')
show = False
IMAGE_DIR = "/home/twsf/data/UnderWater/val/images/"


def plot_img(img, bboxes):
    img = img.astype(np.float64) / 255.0 if img.max() > 1.0 else img
    box_colors = ((1, 0, 0), (0, 1, 0))
    for bbox in bboxes:
        try:
            if -1 in bbox:
                continue
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            id = int(bbox[4])
            # plot
            box_color = box_colors[min(id, len(box_colors)-1)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=3)

        except Exception as e:
            print(e)
            continue

    return img


class DefaultEval(object):
    def __init__(self):
        self.stats = []

    def calc_iou(self, a, b):
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)

        ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
        ua = torch.clamp(ua, min=1e-8)

        intersection = iw * ih
        IoU = intersection / ua

        return IoU

    def statistics(self, prediction, ground_truth, iou_thresh=0.5):
        """
        Arg:
            prediction: result of after use nms, shape like [batch, M, box + cls + score]
            ground_truth: shape like [batch, N, box + cls]
        return:
            stats(list):
                correct: prediction right or wrong, [0, 1, 1, ...], type list
                prediction confident: [], type list
                prediction classes: [], type list
                truth classes: [], type list
        """

        batch_size = len(ground_truth)
        stats = []
        for id in range(batch_size):
            targets = ground_truth[id]  # id'th image gt
            preds = prediction[id]  # id'th image pred
            tcls = targets[:, 4].tolist() if len(targets) else []
            num_gt = len(targets)  # number of target

            # predict is none
            if preds is None:
                # supposing that pred is none and gt is not
                if num_gt > 0:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Assign all predictions as incorrect
            correct = [0] * len(preds)
            if num_gt:
                detected = []
                tcls_tensor = targets[:, 4]

                # target boxes
                tboxes = targets[:, :4]

                for ii, pred in enumerate(preds):
                    pbox = pred[:4].unsqueeze(0)
                    pcls = pred[4]

                    # Break if all targets already located in image
                    if len(detected) == num_gt:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = self.calc_iou(pbox, tboxes[m]).max(1)

                    # If iou > threshold and gt was not matched
                    if iou > iou_thresh and m[bi] not in detected:
                        correct[ii] = 1
                        detected.append(m[bi])

            # (correct, pconf, pcls, tcls)
            stats.append((correct, preds[:, 5].tolist(), preds[:, 4].tolist(), tcls))

        self.stats += stats

    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Calculate area under PR curve, looking for points where x axis (recall) changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # Sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def ap_per_class(self, tp, conf, pred_cls, target_cls):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:    True positives (list).
            conf:  Objectness value from 0-1 (list).
            pred_cls: Predicted object classes (list).
            target_cls: True object classes (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls)

        # Create Precision-Recall curve and compute AP for each class
        ap, p, r = [], [], []
        for c in unique_classes:
            i = pred_cls == c
            n_gt = (target_cls == c).sum()  # Number of ground truth objects
            n_p = i.sum()  # Number of predicted objects

            if n_p == 0 and n_gt == 0:
                continue
            elif n_p == 0 or n_gt == 0:
                ap.append(0)
                r.append(0)
                p.append(0)
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum()
                tpc = (tp[i]).cumsum()

                # Recall
                recall = tpc / (n_gt + 1e-16)  # recall curve
                r.append(recall[-1])

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p.append(precision[-1])

                # AP from recall-precision curve
                ap.append(self.compute_ap(recall, precision))

                # Plot
                # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                # ax.plot(np.concatenate(([0.], recall)), np.concatenate(([0.], precision)))
                # ax.set_xlabel('YOLOv3-SPP')
                # ax.set_xlabel('Recall')
                # ax.set_ylabel('Precision')
                # ax.set_xlim(0, 1)
                # fig.tight_layout()
                # fig.savefig('PR_curve.png', dpi=300)

        # Compute F1 score (harmonic mean of precision and recall)
        p, r, ap = np.array(p), np.array(r), np.array(ap)
        f1 = 2 * p * r / (p + r + 1e-16)

        return p, r, ap, f1, unique_classes.astype('int32')


class DET_toolkit(object):
    def __init__(self):
        self.gt_dir = ANNOTATION_DIR
        self.img_dir = IMAGE_DIR
        self.classes = CLASSES
        self.classes2id = {'holothurian': 0, 'echinus': 1, 'scallop': 2, 'starfish': 3, 'waterweeds': 4}

    def load_anno(self, anno_xml):
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
            bndbox.append(self.classes2id[cls])
            box_all += [bndbox]

        return np.array(box_all).astype(np.float32)

    def __call__(self):
        def_eval = DefaultEval()

        # get val predict box
        with open(CHIP_RESULT_FILE, 'r') as f:
            results = json.load(f)
        with open(LOCATION_FILE, 'r') as f:
            detecions = dict()
        for det in tqdm(results):
            img_id = det['image_id']
            cls_id = det['category_id']
            bbox = det['bbox']
            score = det['score']
            img_name = img_id
            if img_name in detecions:
                detecions[img_name].append(bbox + [score, cls_id])
            else:
                detecions[img_name] = [bbox + [score, cls_id]]

        # metrics
        for img_name, det in tqdm(detecions.items()):
            pred_bbox = nms(det, score_threshold=0.5)[:, [0, 1, 2, 3, 5, 4]].astype(np.float32)
            gt_bbox = self.load_anno(osp.join(self.gt_dir, img_name[:-4]+'.xml'))

            def_eval.statistics(
                torch.tensor(pred_bbox).unsqueeze(0),
                torch.tensor(gt_bbox).unsqueeze(0))

            if show:
                img = cv2.imread(osp.join(self.img_dir, img_name))[:, :, ::-1]
                gt_bbox[:, 4] = 0
                pred_bbox[:, 4] = 1
                img = plot_img(img, gt_bbox)
                img = plot_img(img, pred_bbox)
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 1, 1).imshow(img)
                # plt.subplot(2, 1, 2).imshow(pred_img)
                plt.show()

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in list(zip(*def_eval.stats))]
        # number of targets per class
        nt = np.bincount(stats[3].astype(np.int64), minlength=len(self.classes))
        if len(stats):
            p, r, ap, f1, ap_class = def_eval.ap_per_class(*stats)
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

        # Print and Write results
        title = ('%20s' + '%10s'*5) % ('Class', 'Targets', 'P', 'R', 'mAP', 'F1')
        print(title)
        printline = '%20s' + '%10.3g' * 5
        pf = printline % ('all', nt.sum(), mp, mr, map, mf1)  # print format
        print(pf)
        if len(self.classes) > 1 and len(stats):
            for i, c in enumerate(ap_class):
                pf = printline % (self.classes[c], nt[c], p[i], r[i], ap[i], f1[i])
                print(pf)


if __name__ == '__main__':
    det = DET_toolkit()
    det()
