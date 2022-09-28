# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from model_util_scannet import rotate_aligned_boxes

from model_util_scannet import ScannetDatasetConfig

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64  # 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])



if __name__ == '__main__':

    split_set = 'train'
    MAX_NUM_OBJ = 100
    use_height = True
    num_points = 40000
    use_color = False

    # OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    OBJ_CLASS_IDS = np.array(
                ['bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',
                 'keyboard', 'lamp', 'laptop', 'monitor',
                 'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet', 'wardrobe'])


    data_path = os.path.join(BASE_DIR, '/mnt/sda/szh/scannet/scannet_train_detection_data_22')
    all_scan_names = list(set([os.path.basename(x)[0:12] \
                               for x in os.listdir(data_path) if x.startswith('scene')]))

    if split_set == 'all':
        scan_names = all_scan_names
    elif split_set in ['train', 'val', 'test']:
        split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                                       'scannetv2_{}.txt'.format(split_set))
        with open(split_filenames, 'r') as f:
            scan_names = f.read().splitlines()
            # remove unavailiable scans
        num_scans = len(scan_names)
        scan_names = [sname for sname in scan_names \
                           if sname in all_scan_names]
        print('kept {} scans out of {}'.format(len(scan_names), num_scans))
        num_scans = len(scan_names)
    else:
        print('illegal split name')
    sum_size = np.zeros((22, 3))
    num_bbox = np.zeros((22, 3))
    for idx in range(len(scan_names)) :
        scan_name = scan_names[idx]
        np.load.__defaults__ = (None, True, True, 'ASCII')
        instance_bboxes = np.load(os.path.join(data_path, scan_name) + '_bbox.npy')
        np.load.__defaults__ = (None, False, True, 'ASCII')
        bboxes_size = np.array(instance_bboxes[:, 3:6],dtype=float)

        for i in range(len(instance_bboxes)):
            for j in range(len(OBJ_CLASS_IDS)):
                if instance_bboxes[i][6] == OBJ_CLASS_IDS[j] :
                    num_bbox[j,:] = num_bbox[j,:] + [1,1,1]
                    sum_size[j,:] = sum_size[j,:] + bboxes_size[i,:]

    means_size = sum_size / num_bbox
    print(means_size)
    np.savez("scannet_means_22.npz", means_size)









