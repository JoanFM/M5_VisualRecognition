from glob import glob
import os
import numpy as np
import cv2

from detectron2.structures import BoxMode
from pycocotools import coco

KITTI_CATEGORIES = {
    'Car': 1,
    'Pedestrian': 2,
}

KITTIMOTS_DATA_DIR = '/home/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_TRAIN_IMG = KITTIMOTS_DATA_DIR+'training/image_02'
KITTIMOTS_TEST_IMG = KITTIMOTS_DATA_DIR+'testing/image_02'
KITTIMOTS_TRAIN_LABEL = KITTIMOTS_DATA_DIR+'instances_txt'

class Inference_Dataloader():
    def __init__(self):
        self.train_paths = glob(os.path.join(KITTIMOTS_TRAIN_IMG,'*','*'))
        self.test_paths = glob(os.path.join(KITTIMOTS_TEST_IMG,'*','*'))

    def load_data(self):
        return {
            'train': self.train_paths,
            'test': self.test_paths
        }

class KITTI_Dataloader():
    def __init__(self, split_perc=0.8):
        if not os.path.isdir(KITTIMOTS_TRAIN_IMG):
            raise Exception('The image directory is not correct.')
        if not os.path.isdir(KITTIMOTS_TRAIN_LABEL):
            raise Exception('The masks directory is not correct.')
        label_paths = sorted(glob(KITTI_TRAIN_LABEL+os.sep+'*.txt'))
        self.label_indices = ['{0:04d}'.format(l) for l in range(len(label_paths))]
        split_point = int(len(img_indices)*split_perc)
        self.train_labels = self.label_indices[:split_point]
        self.test_labels = self.label_indices[split_point:]
    
    def get_dicts(self, train_flag=False):
        sequences = self.train_indices if train_flag is True else self.test_indices
        dataset_dicts = []
        for seq in sequences:
            seq_dicts = []
            image_paths = sorted(glob(KITTIMOTS_TRAIN_IMG+os.sep+seq+os.sep+'*.png'))
            num_frames = len(image_paths)
            label_path = KITTIMOTS_TRAIN_LABEL+os.sep+seq+'.txt'
            with open(label_path,'r') as file:
                lines = file.readlines()
                lines = [l.split(' ') for l in lines]
            for k in range(num_frames):
                frame_lines = [l for l in lines if int(l[0]) == k]
                frame_annotations = []
                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                for detection in frame_lines:
                    rle = {
                        'counts': detection[-1].strip(),
                        'size': [h, w]
                    }
                    bbox = coco.maskUtils.toBbox(rle)
                    annotation = {
                        'category_id': int(detection[1])%1000 #detection[2]?
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'bbox':bbox
                    }
                    frame_annotations.append(annotation)
                filename = '{0:06d}.png'.format(k)
                img_dict = {
                    'file_name': os.path.join(KITTIMOTS_TRAIN_IMG,seq,filename),
                    'image_id': k+(int(seq)*1e3),
                    'height': h,
                    'width': w,
                    'annotations': frame_annotations
                }
                seq_dicts.append(img_dict)
            dataset_dicts += seq_dicts
        return dataset_dicts
