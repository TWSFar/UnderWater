import os
import shutil
import os.path as osp
import concurrent.futures
set_path = '/home/twsf/data/UnderWater/ImageSets/Main/val.txt'
img_path = '/home/twsf/data/UnderWater/JPEGImages/{}.jpg'
anno_path = '/home/twsf/data/UnderWater/Annotations/{}.xml'
img_dest_dir = '/home/twsf/data/UnderWater/val/images'
anno_dest_dir = '/home/twsf/data/UnderWater/val/labels'


def _copy(src_image, dest_path):
    shutil.copy(src_image, dest_path)


if __name__ == "__main__":
    img_list = []
    anno_list = []
    with open(set_path, 'r') as f:
        for img_id in f.readlines():
            img_list.append(img_path.format(img_id.strip()))
            anno_list.append(anno_path.format(img_id.strip()))

    print('copy val images....')
    with concurrent.futures.ThreadPoolExecutor() as exector:
        exector.map(_copy, img_list, [img_dest_dir]*len(img_list))

    print('copy val annotations....')
    with concurrent.futures.ThreadPoolExecutor() as exector:
        exector.map(_copy, anno_list, [anno_dest_dir]*len(anno_list))
