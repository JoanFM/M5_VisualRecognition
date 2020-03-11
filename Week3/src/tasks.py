import os
from glob import glob
import numpy as np
import cv2
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from .utils import KITTIMOTS_Inference_Dataloader


SAVE_PATH = './results_week_3'

def KITTIMOTS_inference_task(model_name, model_file):
    path = os.path.join(SAVE_PATH, model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    # Loading training and test examples
    dataloader = KITTIMOTS_Inference_Dataloader()
    dataset = dataloader.load_data()

    # Load model and checkpoint
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    predictor = DefaultPredictor(cfg)

    # Qualitative results: visualize some prediction results on MIT_split dataset
    for i, img_path in enumerate(dataset['test'][:20]):
        img = cv2.imread(img_path)
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=0.8, 
            instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(os.path.join(path, 'Inference_' + model_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])
