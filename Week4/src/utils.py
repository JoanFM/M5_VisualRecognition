import os
from glob import glob
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from PIL import Image
from pycocotools import coco
import torch

from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
from detectron2.structures import BoxMode

KITTIMOTS_SPLIT_POINT = 12
KITTIMOTS_DATA_DIR = '/home/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_TRAIN_IMG = KITTIMOTS_DATA_DIR+'training/image_02'
KITTIMOTS_TRAIN_LABEL = KITTIMOTS_DATA_DIR+'instances_txt'
KITTIMOTS_TRAIN_MASK = KITTIMOTS_DATA_DIR+'instances'
KITTI_CATEGORIES = {
    'Car': 1,
    'Pedestrian': 2,
    'Ignore': 10,
    'Background': 0
}
"""
COCO_CATTEGORIES = {
    1: 2,
    2: 0,
    10: None,
}
"""

class KITTIMOTS_Dataloader():
    def __init__(self):
        if not os.path.isdir(KITTIMOTS_TRAIN_IMG):
            raise Exception('The image directory is not correct.')
        if not os.path.isdir(KITTIMOTS_TRAIN_LABEL):
            raise Exception('The labels directory is not correct.')
        if not os.path.isdir(KITTIMOTS_TRAIN_MASK):
            raise Exception('The masks directory is not correct')
        label_paths = sorted(glob(KITTIMOTS_TRAIN_LABEL+os.sep+'*.txt'))
        label_indices = ['{0:04d}'.format(l) for l in range(len(label_paths))]
        self.train_sequences = label_indices[:KITTIMOTS_SPLIT_POINT]
        self.val_sequences = label_indices[KITTIMOTS_SPLIT_POINT:]
        print(f'Train Sequences: {self.train_sequences}')
        print(f'Validation Sequences: {self.val_sequences}')

    def get_dicts(self, train_flag=False):
        sequences = self.train_sequences if train_flag is True else self.val_sequences
        dataset_dicts = []
        for seq in sequences:
            seq_dicts = []
            image_paths = sorted(glob(KITTIMOTS_TRAIN_IMG+os.sep+seq+os.sep+'*.png'))
            mask_paths = sorted(glob(KITTIMOTS_TRAIN_MASK+os.sep+seq+os.sep+'*.png'))
            num_frames = len(image_paths)
            label_path = KITTIMOTS_TRAIN_LABEL+os.sep+seq+'.txt'
            with open(label_path,'r') as file:
                lines = file.readlines()
                lines = [l.split(' ') for l in lines]
            for k in range(num_frames):
                frame_lines = [l for l in lines if int(l[0]) == k]
                if frame_lines:
                    frame_annotations = []
                    h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                    for detection in frame_lines:
                        rle = detection[-1].strip().encode(encoding='UTF-8')
                        print(rle)
                        segm = {
                            'counts': rle,
                            'size': [h, w]
                        }
                        bbox = coco.maskUtils.toBbox(segm).tolist()
                        print(bbox)
                        bbox[2] += bbox[0]
                        bbox[3] += bbox[1]
                        print(bbox)
                        #bbox = [int(item) for item in bbox]
                        category_id = int(detection[2])

                        """
                        mask = coco.maskUtils.decode(rle)
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        seg = [list(c.flatten()) for c in contours]
                        """

                        annotation = {
                            'category_id': category_id,
                            'bbox_mode': BoxMode.XYXY_ABS,
                            'bbox': bbox,
                            'segmentation': rle,
                            'is_crowd': 0 if category_id in [1,2,0] else 1
                        }
                        frame_annotations.append(annotation)
                    filename = '{0:06d}.png'.format(k)
                    img_path = os.path.join(KITTIMOTS_TRAIN_IMG,seq,filename)
                    mask_path = os.path.join(KITTIMOTS_TRAIN_MASK,seq,filename)
                    segmentation = torch.Tensor(np.array(Image.open(mask_path)))
                    print(segmentation.size)
                    img_dict = {
                        'file_name': img_path,
                        'sem_seg_file_name': mask_path,
                        'sem_seg': segmentation
                        'image_id': k+(int(seq)*1e3),
                        'height': h,
                        'width': w,
                        'annotations': frame_annotations
                    }
                    seq_dicts.append(img_dict)
            dataset_dicts += seq_dicts
        return dataset_dicts