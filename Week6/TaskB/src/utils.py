# -- IMPORTS -- #
from matplotlib import pyplot as plt
import pycocotools.mask as mask_utils
from itertools import groupby
from pycocotools import coco
from random import shuffle
from glob import glob
import numpy as np
import json
import torch
import os
import cv2

from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.structures import BoxMode

# -- CONSTANTS -- #
KITTI_CATEGORIES = {
    'Car': 1,
    'Dummy': 0, # We need 2 classes to not get NANs when evaluating, for some reason, duh
}
COCO_CATEGORIES = {
    1: 2,
}

TRAIN_SEQ = [1,2,6,18,20]
VALID_SEQ = [0,3,10,12,14]
TEST_SEQ = [4,5,7,8,9,11,15]

KITTIMOTS_DATA_DIR = '/home/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_TRAIN_IMG = KITTIMOTS_DATA_DIR + 'training/image_02'
KITTIMOTS_TRAIN_LABEL = KITTIMOTS_DATA_DIR + 'instances_txt'
KITTIMOTS_TRAIN_MASK = KITTIMOTS_DATA_DIR + 'instances'

VIRTUAL_KITTI_DATA_DIR = '/home/mcv/datasets/vKITTI/'
INTERMIDIATE_MASK = '/clone/frames/instanceSegmentation/Camera_0'
INTERMIDIATE_IMG = '/clone/frames/rgb/Camera_0'

# -- UTILS -- #
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def plot_validation_loss(cfg, iterations, model_name, savepath, filename):
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
    plt.savefig(os.path.join(savepath, filename))

# -- DATALOADERS -- #
class KittiMots():
    def __init__(self):
        self.train_img_dir = KITTIMOTS_TRAIN_IMG
        self.train_label_dir = KITTIMOTS_TRAIN_LABEL
        self.train_mask_dir = KITTIMOTS_TRAIN_MASK
        if not os.path.isdir(self.train_img_dir):
            raise Exception('The image directory is not correct.')
        if not os.path.isdir(self.train_label_dir):
            raise Exception('The labels directory is not correct.')
        if not os.path.isdir(self.train_mask_dir):
            raise Exception('The masks directory is not correct')
        self.train_sequences = ['{0:04d}'.format(l) for l in TRAIN_SEQ]
        self.val_sequences = ['{0:04d}'.format(l) for l in VALID_SEQ]
        self.test_sequences = ['{0:04d}'.format(l) for l in TEST_SEQ]
        print(f'Train Sequences: {self.train_sequences}')
        print(f'Validation Sequences: {self.val_sequences}')
        print(f'Test Sequences: {self.test_sequences}')
    
    def get_dicts(self, flag='train', method='complete', percentage=1.0):
        if flag == 'train':
            sequences = self.train_sequences
        elif flag == 'val':
            sequences = self.val_sequences
        elif flag == 'test':
            sequences = self.test_sequences
        else:
            raise ValueError('The flag only accepts "train", "val" or "test".')
        dataset_dicts = []
        for seq in sequences:
            seq_dicts = self.get_seq_dicts(seq)
            dataset_dicts += seq_dicts
        if method == 'complete':
            split_point = len(dataset_dicts)*percentage
            result = dataset_dicts[:split_point]
        elif method == 'random':
            split_point = len(dataset_dicts)*percentage
            result = shuffle(dataset_dicts)[:split_point]
        else:
            raise ValueError('Method only accepts "complete" or "random".')
        return result

    def get_seq_dicts(self, seq):
        image_paths = sorted(glob(os.path.join(self.train_img_dir, seq, '*.png')))
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
            if category_id not in KITTI_CATEGORIES.values():
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
                'category_id': 2,
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': bbox,
                'segmentation': seg,
            }
            frame_annotations.append(annotation)
        return frame_annotations

    def get_img_dict(self, seq, k, h, w, frame_annotations):
        filename = '{0:06d}.png'.format(k)
        img_path = os.path.join(self.train_img_dir,seq,filename)
        img_dict = {
            'file_name': img_path,
            'image_id': k + (int(seq) * 1e3),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }
        return img_dict

class VirtualKitti():
    def __init__(self):
        self.sequences = ['Scene{0:02d}'.format(l) for l in item in TRAIN_SEQ]
        print(f'Train Sequences: {self.sequences}')
    
    def get_dicts(self):
        dataset_dicts = []
        for seq in self.sequences:
            seq_dicts = self.get_seq_dicts(seq)
            dataset_dicts += seq_dicts
        return dataset_dicts
    
    def get_seq_dicts(self, seq):
        image_paths = sorted(glob(VIRTUAL_KITTI_DATA_DIR+seq+INTERMIDIATE_IMG+os.sep+'*.jpg'))
        mask_paths = sorted(glob(VIRTUAL_KITTI_DATA_DIR+seq+INTERMIDIATE_MASK+os.sep+'*.png'))
        seq_dicts = []
        for k, (m_path, i_path) in enumerate(zip(mask_paths,image_paths)):
            img = np.array(Image.open(m_path))
            frame_annotations = self.get_frame_annotations(img)
            img_dict = self.get_img_dict(seq, k, i_path, img, frame_annotations)
            seq_dicts.append(img_dict)
        return seq_dicts
    
    def get_frame_annotations(self, img):
        h, w = img.shape 
        frame_annotations = []
        instances = np.unique(img)
        for ins in instances[1:]:
            mask = (img==ins)/ins
            rle = mask_utils.frPyObjects(binary_mask_to_rle(mask), w, h)
            bbox = coco.maskUtils.toBbox(rle).tolist()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(item) for item in bbox]
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            seg = [[int(i) for i in c.flatten()] for c in contours]
            seg = [s for s in seg if len(s) >= 6]
            if not seg:
                continue
            annotation = {
                'category_id': 2,
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': bbox,
                'segmentation': seg,
            }
            frame_annotations.append(annotation)
        return frame_annotations

    def get_img_dict(self, seq, k, filename, img, frame_annotations):
        h, w = img.shape
        img_dict = {
            'file_name': filename,
            'image_id': k + (int(seq[-2:]) * 1e4),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }
        return img_dict

# -- HOOKS -- #
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