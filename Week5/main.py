from src.tasks import task_a, task_b

def main():
    """
    MASK RCNN Configurations:
        - COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
        - Cityscapes/mask_rcnn_R_50_FPN.yaml

    CHECKPOINTS from previous models from last week:
        - COCO+KITTIMOTS (0.504 AP): "results_week_4_task_b/MaskRCNN_R_101_FPN/model_final.pth"
        - COCO+Cityscapes+KITTIMOTS: "results_week_4_task_b/MaskRCNN_R_50_FPN_Cityscapes/model_final.pth"
    """
    # TASK A:
    task_a("MaskRCNN_R_101_FPN","COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",checkpoint=None,evaluate=True,visualize=True)
    #task_a("MaskRCNN_R_50_FPN_Cityscapes","Cityscapes/mask_rcnn_R_50_FPN.yaml",checkpoint=None,evaluate=True,visualize=True)
    #task_a("MaskRCNN_R_101_FPN","COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",checkpoint="results_week_4_task_b/MaskRCNN_R_101_FPN/model_final.pth",evaluate=True,visualize=True)
    #task_a("MaskRCNN_R_50_FPN_Cityscapes","Cityscapes/mask_rcnn_R_50_FPN.yaml",checkpoint="results_week_4_task_b/MaskRCNN_R_50_FPN_Cityscapes/model_final.pth",evaluate=True,visualize=True)
    # TASK B:
    """
    task_b("MaskRCNN_R_101_FPN","COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",checkpoint=None)
    task_b("MaskRCNN_R_50_FPN_Cityscapes","Cityscapes/mask_rcnn_R_50_FPN.yaml",checkpoint=None)
    task_b("MaskRCNN_R_101_FPN","COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",checkpoint="results_week_4_task_b/MaskRCNN_R_101_FPN/model_final.pth")
    task_b("MaskRCNN_R_50_FPN_Cityscapes","Cityscapes/mask_rcnn_R_50_FPN.yaml",checkpoint="results_week_4_task_b/MaskRCNN_R_50_FPN_Cityscapes/model_final.pth")
    """
if __name__ == '__main__':
    main()
