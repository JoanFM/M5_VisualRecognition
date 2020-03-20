from src.tasks import task_a

def main():
    """ 
    MASK RCNN Configurations:
        - COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml
        - COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
    """

    # -- TASK A -- #
    # task_a("MaskRCNN_R_50_C4", "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")
    # task_a("MaskRCNN_R_50_DC5", "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
    # task_a("MaskRCNN_R_50_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # task_a("MaskRCNN_R_101_C4", "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")
    # task_a("MaskRCNN_R_101_DC5", "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml")
    # task_a("MaskRCNN_R_101_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

    
if __name__ == '__main__':
    main()
