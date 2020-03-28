import os
import numpy as np
import cv2
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from .utils import MOTS_Dataloader
from .utils import KITTI_CATEGORIES, MOTS_CATEGORIES
from .utils import ValidationLoss, plot_validation_loss


def task_a(model_name, model_file, checkpoint=None, evaluate=True, visualize=True):
    print('Running task A for model', model_name)
    if checkpoint:
        SAVE_PATH = os.path.join('./results_week_5_task_a', model_name + '_wCheckpoint')
    else:
        SAVE_PATH = os.path.join('./results_week_5_task_a', model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Loading data
    print('Loading data')
    dataloader = MOTS_Dataloader(dataset='motschallenge')
    def mots_train(): return dataloader.get_dicts(train_flag=True)
    def mots_val(): return dataloader.get_dicts(train_flag=False)
    DatasetCatalog.register('MOTS_train', mots_train)
    MetadataCatalog.get('MOTS_train').set(thing_classes=list(MOTS_CATEGORIES.keys()))
    DatasetCatalog.register('MOTS_val', mots_val)
    MetadataCatalog.get('MOTS_val').set(thing_classes=list(MOTS_CATEGORIES.keys()))

    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    model_training_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) # Store current model training metadata
    cfg.DATASETS.TRAIN = ('MOTS_train', )
    cfg.DATASETS.TEST = ('MOTS_val', )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH
    if checkpoint:
        print('Using Checkpoint')
        cfg.MODEL.WEIGHTS = checkpoint
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    
    if evaluate:
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

        # Evaluation
        print('Evaluating')
        evaluator = COCOEvaluator('MOTS_val', cfg, False, output_dir=SAVE_PATH)
        trainer = DefaultTrainer(cfg)
        trainer.test(cfg, model, evaluators=[evaluator])

    if visualize:
        # Qualitative results: visualize some results
        print('Getting qualitative results')
        predictor = DefaultPredictor(cfg)
        inputs = mots_val()
        inputs = inputs[:20] + inputs[-20:]
        for i, input in enumerate(inputs):
            img = cv2.imread(input['file_name'])
            outputs = predictor(img)
            v = Visualizer(
                img[:, :, ::-1],
                metadata=model_training_metadata,
                scale=0.8,
                instance_mode=ColorMode.IMAGE)
            v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            cv2.imwrite(os.path.join(SAVE_PATH, 'Inference_' + model_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])

def task_b(model_name, model_file, checkpoint=None):
    print('Running task B for model', model_name)
    if checkpoint:
        SAVE_PATH = os.path.join('./results_week_5_task_b', model_name + '_wCheckpoint')
    else:
        SAVE_PATH = os.path.join('./results_week_5_task_b', model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)
    kitti_val, _ = loading_data()
    # DEFAULT PARAMETERS
    hyperparams = {
        'lr': 0.00025,
        'batch': 4,
        'scheduler': 'WarmupMultiStepLR',
        'iou': [0.3,0.7],
        'top_k_train': 12000
    }
    training_loop(SAVE_PATH, model_name, model_file, hyperparams, kitti_val, checkpoint=checkpoint, visualize=True)
    
def task_c(model_name, model_file, checkpoint=None):
    print('Running task C for model', model_name)
    SAVE_PATH = os.path.join('./results_week_5_task_c', model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)
    kitti_val, _ = loading_data()
    loop = get_hyper_params()
    for hyperparams in loop:
        print('Running experiment with params:')
        for key, value in hyperparams.items():
            print('{0}: {1}'.format(key,value))
        training_loop(SAVE_PATH, model_name, model_file, hyperparams, kitti_val, checkpoint=checkpoint, visualize=False)

def loading_data():
    # Loading data
    print('Loading data')
    motsloader = MOTS_Dataloader(dataset='motschallenge')
    kittiloader = MOTS_Dataloader(dataset='kittimots')
    def mots_train(): return motsloader.get_dicts(train_flag=True)
    def kitti_val(): return kittiloader.get_dicts(train_flag=False)
    DatasetCatalog.register('MOTS_train', mots_train)
    MetadataCatalog.get('MOTS_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTIMOTS_val', kitti_val)
    MetadataCatalog.get('KITTIMOTS_val').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    return kitti_val, mots_train

def training_loop(SAVE_PATH, model_name, model_file, hyperparams, dataloader, checkpoint=None, visualize=True):
    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ('MOTS_train', )
    cfg.DATASETS.TEST = ('KITTIMOTS_val', )
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH
    if checkpoint:
        last_checkpoint = torch.load(checkpoint)
        new_path = checkpoint.split('.')[0]+'_modified.pth'
        last_checkpoint['iteration'] = -1
        torch.save(last_checkpoint,new_path)
        cfg.MODEL.WEIGHTS = new_path
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    cfg.SOLVER.IMS_PER_BATCH = hyperparams['batch']
    cfg.SOLVER.BASE_LR = hyperparams['lr']
    cfg.SOLVER.LR_SCHEDULER_NAME = hyperparams['scheduler']
    cfg.MODEL.RPN.IOU_THRESHOLDS = hyperparams['iou']
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = hyperparams['top_k_train']
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.SCORE_THRESH = 0.5

    # Training
    print('Training')
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation
    print('Evaluating')
    evaluator = COCOEvaluator('KITTIMOTS_val', cfg, False, output_dir=SAVE_PATH)
    trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    print('Plotting losses')
    plot_validation_loss(cfg, cfg.SOLVER.MAX_ITER, model_name, SAVE_PATH)

    if visualize:
        # Qualitative results: visualize some results
        print('Getting qualitative results')
        predictor = DefaultPredictor(cfg)
        predictor.model.load_state_dict(trainer.model.state_dict())
        def kitti_val(): return dataloader.get_dicts(train_flag=False)
        inputs = kitti_val()
        inputs = inputs[:20] + inputs[-20:]
        for i, input in enumerate(inputs):
            file_name = input['file_name']
            print('Prediction on image ' + file_name)
            img = cv2.imread(file_name)
            outputs = predictor(img)
            v = Visualizer(
                img[:, :, ::-1],
                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                scale=0.8,
                instance_mode=ColorMode.IMAGE)
            v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            cv2.imwrite(os.path.join(SAVE_PATH, 'Inference_' + model_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])

def get_hyper_params():
    lr = [0.0025, 0.0001, 0.00025, 0.0005]
    batch = [4, 8, 16]
    scheduler = ['WarmupMultiStepLR','WarmupCosineLR']
    iou = [[0.2, 0.8],[0.3, 0.7],[0.4, 0.6]]
    top_k_train = [6000, 9000, 12000, 15000]
    loop = []
    for value in lr:
        loop.append({ 'lr': value, 'batch': 4, 'scheduler': 'WarmupMultiStepLR', 'iou': [0.3,0.7], 'top_k_train': 12000})
    for value in batch:
        loop.append({ 'lr': 0.00025, 'batch': value, 'scheduler': 'WarmupMultiStepLR', 'iou': [0.3,0.7], 'top_k_train': 12000})
    for value in scheduler:
        loop.append({ 'lr': 0.00025, 'batch': 4, 'scheduler': value, 'iou': [0.3,0.7], 'top_k_train': 12000})
    for value in iou:
        loop.append({ 'lr': 0.00025, 'batch': 4, 'scheduler': 'WarmupMultiStepLR', 'iou': value, 'top_k_train': 12000})
    for value in batch:
        loop.append({ 'lr': 0.00025, 'batch': 4, 'scheduler': 'WarmupMultiStepLR', 'iou': [0.3,0.7], 'top_k_train': value})
    return loop