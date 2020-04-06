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

from .utils import KittiMots, VirtualKitti
from .utils import KITTI_CATEGORIES
from .utils import ValidationLoss, plot_validation_loss


def experiment_4(exp_name, model_file, method):

    print('Running Task B experiment', exp_name)
    SAVE_PATH = os.path.join('./results_week_6', exp_name+'_'+method)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Loading data
    print('Loading data')
    virtualoader = VirtualKitti()
    kittiloader = KittiMots()
    def rkitti_val(): return kittiloader.get_dicts(flag='val')
    def rkitti_test(): return kittiloader.get_dicts(flag='test')
    DatasetCatalog.register('KITTI_val', rkitti_val)
    MetadataCatalog.get('KITTI_val').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTI_test', rkitti_test)
    MetadataCatalog.get('KITTI_test').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    for per in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]:
        print('Iteration 100% Virtual & {0}% Real'.format(per*100))
        if os.path.isfile(os.path.join(SAVE_PATH, 'metrics.json')):
            os.remove(os.path.join(SAVE_PATH, 'metrics.json'))
        def vkitti_train():
            virtual = virtualoader.get_dicts()
            real = kittiloader.get_dicts(flag='train', method=method, percentage=per)
            all_data = virtual + real
            return all_data
        catalog_name = 'ALL_train_{0}'.format(int(per*10))
        DatasetCatalog.register(catalog_name, vkitti_train)
        MetadataCatalog.get(catalog_name).set(thing_classes=list(KITTI_CATEGORIES.keys()))

        # Load model and configuration
        print('Loading Model')
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_file))
        cfg.DATASETS.TRAIN = (catalog_name, )
        cfg.DATASETS.TEST = ('KITTI_val', )
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.OUTPUT_DIR = SAVE_PATH
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
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
        filename = 'validation_loss_{0}.png'.format(int(per*10))
        plot_validation_loss(cfg, cfg.SOLVER.MAX_ITER, exp_name, SAVE_PATH, filename)

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
            os.makedirs(os.path.join(SAVE_PATH, str(int(per*10))), exist_ok=True)
            cv2.imwrite(os.path.join(SAVE_PATH, str(int(per*10)), 'Inference_' + exp_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])