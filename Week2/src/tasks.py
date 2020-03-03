import os
from glob import glob
import numpy as np
import cv2
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from utils import dataloader


DATASET_PATH = '/home/grupo07/MIT_split'
SAVE_PATH = './results'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def evaluate(cfg):
    # Quantitative results: compute AP
    trainer = DefaultTrainer(cfg)
    evaluator = COCOEvaluator("MIT_split_test", cfg, False)
    val_loader = build_detection_test_loader(cfg, "MIT_split_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)


def inference_task(model_name, model_file):
    # TODO: Load dataset
    DL = dataloader()
    dataset = DL.load_data()

    # Load model and checkpoint
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url()
    cfg.DATASETS.TEST = ("MIT_split_test")
    predictor = DefaultPredictor(cfg)

    # Qualitative results: visualize some prediction results on MIT_split dataset
    for i, img_path in enumerate(random.sample(dataset["test"], 3)):
        img = cv2.imread(img_path)
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=metadata
            scale=0.8, 
            instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(SAVE_PATH, 'Inference_' + model_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])

