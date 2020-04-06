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

from .utils import KittiMots
from .utils import KITTI_CATEGORIES
from .utils import ValidationLoss, plot_validation_loss


def experiment_3(exp_name, model_file, checkpoint=None):

    print('Running Task B experiment', exp_name)
    SAVE_PATH = os.path.join('./results_week_6', exp_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Loading data
    print('Loading data')
    kittiloader = KittiMots()
    def rkitti_train(): return kittiloader.get_dicts(flag='train', method='complete', percentage= 1.0)
    def rkitti_val(): return kittiloader.get_dicts(flag='val')
    def rkitti_test(): return kittiloader.get_dicts(flag='test')
    DatasetCatalog.register('KITTI_train', vkitti_train)
    MetadataCatalog.get('KITTI_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTI_val', rkitti_val)
    MetadataCatalog.get('KITTI_val').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTI_test', rkitti_test)
    MetadataCatalog.get('KITTI_test').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ('KITTI_train', )
    cfg.DATASETS.TEST = ('KITTI_val', )
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
        raise ValueError('You forgot to put the chekpoint for this experiment')
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupMultiStepLR'
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.2,0.8]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
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
    evaluator = COCOEvaluator('KITTI_test', cfg, False, output_dir=SAVE_PATH)
    trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    print('Plotting losses')
    plot_validation_loss(cfg, cfg.SOLVER.MAX_ITER, exp_name, SAVE_PATH, 'validation_loss.png')

    # Qualitative results: visualize some results
    print('Getting qualitative results')
    predictor = DefaultPredictor(cfg)
    predictor.model.load_state_dict(trainer.model.state_dict())
    inputs = rkitti_test()
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
        cv2.imwrite(os.path.join(SAVE_PATH, 'Inference_' + exp_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])