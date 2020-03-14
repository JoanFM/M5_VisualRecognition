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

from .utils import ValidationLoss
from .utils import KITTIMOTS_Dataloader, Inference_Dataloader
from .utils import KITTI_CATEGORIES

SAVE_PATH = './results_week_3_task_C'

def KITTIMOTS_inference_task(model_name, model_file):
    path = os.path.join(SAVE_PATH, model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    # Loading training and test examples
    print('Loading Data...')
    dataloader = Inference_Dataloader()
    dataset = dataloader.load_data()

    # Load model and checkpoint
    print('Loading Model and Checkpoint..')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    predictor = DefaultPredictor(cfg)

    # Qualitative results: visualize some prediction results on MIT_split dataset
    print('Getting Qualitative Results...')
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

def KITTIMOTS_evaluation_task(model_name, model_file):
    path = os.path.join(SAVE_PATH, 'eval_inf_task', model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    # Load Data
    print('Loading Data.')
    dataloader = KITTIMOTS_Dataloader()
    def kitti_test(): return dataloader.get_dicts(train_flag=False)
    DatasetCatalog.register("KITTIMOTS_test", kitti_test)
    MetadataCatalog.get("KITTIMOTS_test").set(thing_classes=[k for k,_ in KITTI_CATEGORIES.items()])

    # Load MODEL and Configuration
    print('Loading Model.')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)

    # Evaluation
    print('Evaluating......')
    evaluator = COCOEvaluator('KITTIMOTS_test', cfg, False, output_dir="./output/")
    
def KITTIMOTS_training_and_evaluation_task(model_name,model_file):
    path = os.path.join(SAVE_PATH, 'train_task', model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    # Load Data
    print('Loading Data.')
    dataloader = KITTIMOTS_Dataloader()
    def kittmots_train(): return dataloader.get_dicts(train_flag=True)
    def kittimots_test(): return dataloader.get_dicts(train_flag=False)
    DatasetCatalog.register("KITTIMOTS_train", kittmots_train)
    MetadataCatalog.get("KITTIMOTS_train").set(thing_classes=[k for k,_ in KITTI_CATEGORIES.items()])
    DatasetCatalog.register("KITTIMOTS_test", kittimots_test)
    MetadataCatalog.get("KITTIMOTS_test").set(thing_classes=[k for k,_ in KITTI_CATEGORIES.items()])

    # PARAMETERS
    print('Loading Model.')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ('KITTIMOTS_train',)
    cfg.DATASETS.TEST = ('KITTIMOTS_test',)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = NUM_IMGS // cfg.SOLVER.IMS_PER_BATCH + 1 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # Training
    print('Training....')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    # EVALUATION
    print('Evaluating....')
    evaluator = COCOEvaluator("KITTIMOTS_test", cfg, False, output_dir="./output/")
    trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    
    # Qualitative results
    print('Inference on trained model')
    predictor = DefaultPredictor(cfg)
    predictor.model.load_state_dict(trainer.model.state_dict())
    dataloader = Inference_Dataloader()
    dataset = dataloader.load_data()
    print('Getting Qualitative Results...')
    for i, img_path in enumerate(dataset['test'][:20]):
        img = cv2.imread(img_path)
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=0.8, 
            instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(os.path.join(path, 'Inference_' + model_name + '_trained_' + str(i) + '.png'), v.get_image()[:, :, ::-1])
