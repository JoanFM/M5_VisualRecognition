import os
import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from .utils import KITTIMOTS_Dataloader
from .utils import KITTI_CATEGORIES
from .utils import ValidationLoss, plot_validation_loss


SAVE_PATH = './results'


def task_a(model_name, model_file, evaluate=True, visualize=True):
    print('Running task A for model', model_name)

    path = os.path.join(SAVE_PATH+'_week_4_task_a', model_name)
    os.makedirs(path, exist_ok=True)

    # Loading data
    print('Loading data')
    dataloader = KITTIMOTS_Dataloader()
    def kitti_train(): return dataloader.get_dicts(train_flag=True)
    def kitti_val(): return dataloader.get_dicts(train_flag=False)
    DatasetCatalog.register('KITTIMOTS_train', kitti_train)
    MetadataCatalog.get('KITTIMOTS_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTIMOTS_val', kitti_val)
    MetadataCatalog.get('KITTIMOTS_val').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    model_training_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) # Store current model training metadata
    cfg.DATASETS.TRAIN = ('KITTIMOTS_train', )
    cfg.DATASETS.TEST = ('KITTIMOTS_val', )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)

    if evaluate:
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

        # Evaluation
        print('Evaluating')
        evaluator = COCOEvaluator('KITTIMOTS_val', cfg, False, output_dir='./output')
        trainer = DefaultTrainer(cfg)
        trainer.test(cfg, model, evaluators=[evaluator])

    if visualize:
        # Qualitative results: visualize some results
        print('Getting qualitative results')
        predictor = DefaultPredictor(cfg)
        inputs = kitti_val()
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
            cv2.imwrite(os.path.join(path, 'Inference_' + model_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])

def task_b(model_name, model_file):
    print('Running task B for model', model_name)

    path = os.path.join(SAVE_PATH+'_week_4_task_b', model_name)
    os.makedirs(path, exist_ok=True)

    # Loading data
    print('Loading data')
    dataloader = KITTIMOTS_Dataloader()
    def kitti_train(): return dataloader.get_dicts(train_flag=True)
    def kitti_val(): return dataloader.get_dicts(train_flag=False)
    DatasetCatalog.register('KITTIMOTS_train', kitti_train)
    MetadataCatalog.get('KITTIMOTS_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTIMOTS_val', kitti_val)
    MetadataCatalog.get('KITTIMOTS_val').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ('KITTIMOTS_train',)
    cfg.DATASETS.TEST = ('KITTIMOTS_val',)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

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
    evaluator = COCOEvaluator('KITTIMOTS_val', cfg, False, output_dir='./output')
    trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    plot_validation_loss(cfg)

    # Qualitative results: visualize some results
    print('Getting qualitative results')
    predictor = DefaultPredictor(cfg)
    inputs = kitti_val()
    inputs = inputs[:20] + inputs[-20:]
    for i, input in enumerate(inputs):
        img = cv2.imread(input['file_name'])
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=0.8,
            instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(os.path.join(path, 'Inference_' + model_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])
