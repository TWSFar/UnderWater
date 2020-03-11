from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import os.path as osp
from tqdm import tqdm
import numpy as np


test_dir = '/home/twsf/data/UnderWater/test/images'
result_file = 'tools_uw/submit.csv'
config_file = '/home/twsf/work/underwater/mmdetection/tools_uw/config_new.py'
checkpoint_file = '/home/twsf/work/underwater/mmdetection/tools_uw/work_dirs/map47_5/epoch_30.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
thd_range = [0.1, 0.07, 0.05, 0.05]
img_list = os.listdir(test_dir)
with open(result_file, 'w') as f:
    f.writelines("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    for i, img_name in enumerate(tqdm(img_list)):
        img_id = osp.splitext(img_name)[0] + '.xml'
        img_path = osp.join(test_dir, img_name)
        result = inference_detector(model, img_path)
        for i, boxes in enumerate(result):
            for j, box in enumerate(boxes):
                if box[4] < thd_range[i]: continue
                f.writelines(model.CLASSES[i]+','+img_id+','+str(box[4]))
                for v in np.round(box[:4]):
                    f.writelines(','+str(int(v)))
                f.writelines('\n')
        # show_result(img_path, result, model.CLASSES)


# result = inference_detector(model, img)
# # visualize the results in a new window
# show_result(img, result, model.CLASSES)
# # or save the visualization results to image files
# show_result(img, result, model.CLASSES, out_file='result.jpg')
