import os
from glob import glob
import numpy as np
import cv2
import random
import pickle

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from .utils import KITTIMOTS_Dataloader
from .utils import KITTI_CATEGORIES

SAVE_PATH = './results'

def task_a(model_name, model_file):
    path = os.path.join(SAVE_PATH+'_task_a', model_name)
    os.makedirs(path, exist_ok=True)

    # Loading validation set
    print('Loading data')
    dataloader = KITTIMOTS_Dataloader()
    def kitti_val(): return dataloader.get_dicts(train_flag=False)
    DatasetCatalog.register('KITTIMOTS_val', kitti_val)
    MetadataCatalog.get('KITTIMOTS_val').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    # Load MODEL and Configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    predictor = DefaultPredictor(cfg)

    predictions_path = os.path.join(SAVE_PATH, 'predictions.pkl')
    if os.path.exists(predictions_path):
        print('Loading old predictions')
        predictions = pickle.load(open(predictions_path, 'rb'))
    else:
        print('Using Model to predict on input')
        predictions = []
        for i, input in enumerate(kitti_val()):
            img_path = input['file_name']
            img = cv2.imread(img_path)
            prediction = predictor(img)
            predictions.append(prediction)
        pickle.dump(predictions, open(predictions_path, 'wb'))

    print('Predictions length ' + str(len(predictions)))
    print('Inputs length ' + str(len(kitti_val())))

    # Evaluation
    print('Evaluating')
    evaluator = COCOEvaluator('KITTIMOTS_val', cfg, False, output_dir='./output/')
    evaluator.reset()
    evaluator.process(kitti_val(), predictions)
    evaluator.evaluate()
