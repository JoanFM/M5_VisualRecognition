import os
from glob import glob
import json
from copy import deepcopy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pycocotools import coco
import torch

from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.structures import BoxMode


MOTSCHALLENGE_DATA_DIR = '/home/mcv/datasets/MOTSChallenge/train/'
MOTSCHALLENGE_TRAIN_IMG = MOTSCHALLENGE_DATA_DIR + 'images'
MOTSCHALLENGE_TRAIN_LABEL = MOTSCHALLENGE_DATA_DIR + 'instances_txt'
MOTSCHALLENGE_TRAIN_MASK = MOTSCHALLENGE_DATA_DIR + 'instances'

KITTIMOTS_SPLIT_POINT = 12
KITTIMOTS_DATA_DIR = '/home/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_TRAIN_IMG = KITTIMOTS_DATA_DIR + 'training/image_02'
KITTIMOTS_TRAIN_LABEL = KITTIMOTS_DATA_DIR + 'instances_txt'
KITTIMOTS_TRAIN_MASK = KITTIMOTS_DATA_DIR + 'instances'

KITTI_CATEGORIES = {
    'Pedestrian': 1,
    'Dummy': 0, # We need 3 classes to not get NANs when evaluating, for some reason, duh
    'Car': 2
}
COCO_CATEGORIES_FROM_KITTI = {
    1: 2,
    2: 0
}
MOTS_CATEGORIES = {
    'Pedestrian': 2
}
COCO_CATEGORIES_FROM_MOTS = {
    2: 0
}


class MOTS_Dataloader():

    def __init__(self, dataset='kittimots'):
        self.dataset = dataset
        if dataset == 'kittimots':
            self.train_img_dir = KITTIMOTS_TRAIN_IMG
            self.train_label_dir = KITTIMOTS_TRAIN_LABEL
            self.train_mask_dir = KITTIMOTS_TRAIN_MASK
        elif dataset == 'motschallenge':
            self.train_img_dir = MOTSCHALLENGE_TRAIN_IMG
            self.train_label_dir = MOTSCHALLENGE_TRAIN_LABEL
            self.train_mask_dir = MOTSCHALLENGE_TRAIN_MASK
        else:
            raise ValueError('Dataset must be "kittimots" or "motschallenge".')

        if not os.path.isdir(self.train_img_dir):
            raise Exception('The image directory is not correct.')
        if not os.path.isdir(self.train_label_dir):
            raise Exception('The labels directory is not correct.')
        if not os.path.isdir(self.train_mask_dir):
            raise Exception('The masks directory is not correct')

        label_paths = sorted(glob(os.path.join(self.train_label_dir, '*.txt')))
        if dataset == 'kittimots':
            label_indices = ['{0:04d}'.format(l) for l in range(len(label_paths))]
            self.train_sequences = label_indices[:KITTIMOTS_SPLIT_POINT]
            self.val_sequences = label_indices[KITTIMOTS_SPLIT_POINT:]
        elif dataset == 'motschallenge':
            label_indices = [item.split('/')[-1][:-4] for item in label_paths]
            self.train_sequences = label_indices
            self.val_sequences = label_indices
        
        print(f'Train Sequences: {self.train_sequences}')
        print(f'Validation Sequences: {self.val_sequences}')

    def get_dicts(self, train_flag=False):
        sequences = self.train_sequences if train_flag is True else self.val_sequences

        dataset_dicts = []
        for seq in sequences:
            seq_dicts = self.get_seq_dicts(seq)
            dataset_dicts += seq_dicts

        return dataset_dicts

    def get_seq_dicts(self, seq):
        image_paths = sorted(glob(os.path.join(self.train_img_dir, seq, '*.png')))
        if not image_paths:
            self.extension_flag = False
            image_paths = sorted(glob(os.path.join(self.train_img_dir, seq, '*.jpg')))
        else:
            self.extension_flag = True
        mask_paths = sorted(glob(os.path.join(self.train_mask_dir, seq, '*.png')))

        label_path = os.path.join(self.train_label_dir, seq + '.txt')
        with open(label_path, 'r') as file:
            lines = file.readlines()
            lines = [l.split(' ') for l in lines]

        seq_dicts = []
        for k in range(len(image_paths)):
            frame_lines = [l for l in lines if int(l[0]) == k]
            if frame_lines:
                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                frame_annotations = self.get_frame_annotations(frame_lines, h, w)
                img_dict = self.get_img_dict(seq, k, h, w, frame_annotations)
                seq_dicts.append(img_dict)

        return seq_dicts

    def get_frame_annotations(self, frame_lines, h, w):
        frame_annotations = []
        for detection in frame_lines:
            category_id = int(detection[2])
            if self.dataset == 'kittimots' and category_id not in KITTI_CATEGORIES.values() or \
                    self.dataset == 'motschallenge' and category_id not in MOTS_CATEGORIES.values():
                continue

            rle = {
                'counts': detection[-1].strip(),
                'size': [h, w]
            }
            bbox = coco.maskUtils.toBbox(rle).tolist()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(item) for item in bbox]

            mask = coco.maskUtils.decode(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            seg = [[int(i) for i in c.flatten()] for c in contours]
            seg = [s for s in seg if len(s) >= 6]
            if not seg:
                continue

            annotation = {
                'category_id': COCO_CATEGORIES_FROM_KITTI[category_id] if self.dataset == 'kittimots' \
                               else COCO_CATEGORIES_FROM_MOTS[category_id],
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': bbox,
                'segmentation': seg,
            }
            frame_annotations.append(annotation)

        return frame_annotations

    def get_img_dict(self, seq, k, h, w, frame_annotations):
        if self.extension_flag:
            filename = '{0:06d}.png'.format(k)
        else:
            filename = '{0:06d}.jpg'.format(k)
        img_path = os.path.join(self.train_img_dir,seq,filename)
        img_dict = {
            'file_name': img_path,
            'image_id': k + (int(seq) * 1e3),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }

        return img_dict


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
                    self.weights = deepcopy(self.trainer.model.state_dict())


def plot_validation_loss(cfg, iterations, model_name, savepath):
    val_loss = []
    train_loss = []
    for line in open(os.path.join(cfg.OUTPUT_DIR, 'metrics.json'), 'r'):
        result = json.loads(line)
        if 'total_val_loss' in result.keys() and 'total_loss' in result.keys():
            val_loss.append(result['total_val_loss'])
            train_loss.append(result['total_loss'])
    val_idx = [int(item) for item in list(np.linspace(0, iterations, len(val_loss)))]
    train_idx = [int(item) for item in list(np.linspace(0, iterations, len(train_loss)))]

    plt.figure(figsize=(10, 10))
    plt.plot(val_idx,val_loss, label='Validation Loss')
    plt.plot(train_idx,train_loss, label='Training Loss')
    plt.title('Validation Loss for model ' + '{0}'.format(model_name))
    plt.xlabel('Iterations')
    plt.ylabel('Validation Loss')
    plt.grid('True')
    plt.legend()
    plt.savefig(os.path.join(savepath, 'validation_loss.png'))
