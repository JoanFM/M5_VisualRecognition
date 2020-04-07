from copy import deepcopy

import numpy as np
import torch

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils


class CustomTrainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=CustomMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomMapper(cfg, True))


class CustomMapper(DatasetMapper):

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.cfg = deepcopy(cfg)
        self.da = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get('da')
        self.tfm_gens = []

    def __call__(self, dataset_dict):

        self.tfm_gens = []

        dataset_dict = deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.is_train:
            # Crop
            if 'crop' in self.da.keys():
                crop_gen = T.RandomCrop(self.da['crop']['type'], self.da['crop']['size'])
                self.tfm_gens.append(crop_gen)
            # Horizontal flip
            if 'flip' in self.da.keys():
                flip_gen = T.RandomFlip(prob=self.da['flip']['prob'], horizontal=self.da['flip']['horizontal'], vertical=self.da['flip']['vertical'])
                self.tfm_gens.append(flip_gen)
  
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
