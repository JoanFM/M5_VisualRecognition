from glob import glob
import os
import numpy as np
import cv2

from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.structures import BoxMode
from pycocotools import coco

KITTI_CATEGORIES = {
    'Car': 10,
    'Pedestrian': 2,
}

KITTIMOTS_DATA_DIR = '/home/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_TRAIN_IMG = KITTIMOTS_DATA_DIR+'training/image_02'
KITTIMOTS_TEST_IMG = KITTIMOTS_DATA_DIR+'testing/image_02'
KITTIMOTS_TRAIN_LABEL = KITTIMOTS_DATA_DIR+'instances_txt'

class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.best_loss = float('inf')
        self.weights = None
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)
                if losses_reduced < self.best_loss:
                    self.best_loss = losses_reduced
                    self.weights = copy.deepcopy(self.trainer.model.state_dict())

class Inference_Dataloader():
    def __init__(self):
        self.train_paths = glob(os.path.join(KITTIMOTS_TRAIN_IMG,'*','*'))
        self.test_paths = glob(os.path.join(KITTIMOTS_TEST_IMG,'*','*'))

    def load_data(self):
        return {
            'train': self.train_paths,
            'test': self.test_paths
        }

class KITTIMOTS_Dataloader():
    def __init__(self, split_perc=0.8):
        if not os.path.isdir(KITTIMOTS_TRAIN_IMG):
            raise Exception('The image directory is not correct.')
        if not os.path.isdir(KITTIMOTS_TRAIN_LABEL):
            raise Exception('The masks directory is not correct.')
        label_paths = sorted(glob(KITTIMOTS_TRAIN_LABEL+os.sep+'*.txt'))
        label_indices = ['{0:04d}'.format(l) for l in range(len(label_paths))]
        split_point = int(len(label_indices)*split_perc)
        self.train_labels = label_indices[:split_point]
        self.test_labels = label_indices[split_point:]
    
    def get_dicts(self, train_flag=False):
        sequences = self.train_labels if train_flag is True else self.test_labels
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
                if frame_lines:
                    frame_annotations = []
                    h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                    for detection in frame_lines:
                        rle = {
                            'counts': detection[-1].strip(),
                            'size': [h, w]
                        }
                        bbox = coco.maskUtils.toBbox(rle)
                        bbox = [int(item) for item in bbox]
                        annotation = {
                            'category_id': int(detection[1])//1000, #detection[2]?
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
