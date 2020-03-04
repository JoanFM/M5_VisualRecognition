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

from .utils import Inference_Dataloader, Train_KITTI_Dataloader, Test_KITTI_Dataloader
from .utils import MIT_DATA_DIR, CATEGORIES


DATASET_PATH = '/home/grupo07/MIT_split'
SAVE_PATH = './results'

def evaluate(cfg):
    # Quantitative results: compute AP
    trainer = DefaultTrainer(cfg)
    evaluator = COCOEvaluator('MIT_split_test', cfg, False)
    val_loader = build_detection_test_loader(cfg, 'MIT_split_test')
    inference_on_dataset(trainer.model, val_loader, evaluator)


def inference_task(model_name, model_file):
    path = os.path.join(SAVE_PATH, model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    # Loading training and test examples
    dataloader = Inference_Dataloader(MIT_DATA_DIR)
    dataset = dataloader.load_data()

    # Load model and checkpoint
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    predictor = DefaultPredictor(cfg)

    # Qualitative results: visualize some prediction results on MIT_split dataset
    for i, img_path in enumerate([i for i in dataset['test'] if 'inside_city' in i][:20]):
        img = cv2.imread(img_path)
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=0.8, 
            instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(os.path.join(path, 'Inference_' + model_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])

def train_task(model_name, model_file):
    #TODO: Finish the details (EVALUATION ETC)
    path = os.path.join(SAVE_PATH, 'train_task', model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    # Load Data
    dataloader = Train_KITTI_Dataloader()
    test_loader = Test_KITTI_Dataloader()
    DatasetCatalog.register('KITTI_train', dataloader.load_data)
    MetadataCatalog.get('KITTI_train').set(thing_classes=list(CATEGORIES.keys()))
    mit_metadata = MetadataCatalog.get('KITTI_train')

    # Load MODEL and configure train hyperparameters
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ('KITTI_train',)
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.DATALOADER.NUM_WORKERS = 1
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10 # 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9

    # TRAIN!!
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # EVAL
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = ('KITTI_test',)
    predictor = DefaultPredictor(cfg)
    dataset_dicts = test_loader.load_data()
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d['file_name'])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(os.path.join(path, 'Inference_' + model_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])

    evaluator = COCOEvaluator('KITTI_test', cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, 'KITTI_test')
    inference_on_dataset(trainer.model, val_loader, evaluator)