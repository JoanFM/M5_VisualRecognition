from src.experiment_1 import experiment_1
from src.experiment_2 import experiment_2
from src.experiment_3 import experiment_3
from src.experiment_4 import experiment_4

"""
    MASK RCNN Configurations:
        - COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
        - Cityscapes/mask_rcnn_R_50_FPN.yaml

    CHECKPOINTS from virtual kitti:
        - 
    """

if __name__ == "__main__":
    #experiment_1('exp1','COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    #experiment_2('exp2','COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    experiment_3('exp3','COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',checkpoint='results_week_6_task_b/exp2/model_final.pth')
    #experiment_4('exp4','COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml','complete')
    #experiment_4('exp4','COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml','random')