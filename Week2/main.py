from src.tasks import inference_task


def main():
    #TODO: try different configurations? Maybe only 2
    """ 
    FASTER RCNN Configurations:
        - COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
        - COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
        - COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
        - COCO-Detection/faster_rcnn_R_50_C4_3x.yaml

    RETINA Configuration:
        - COCO-Detection/retinanet_R_50_FPN_3x.yaml
        - COCO-Detection/retinanet_R_101_FPN_3x.yaml
    """
    inference_task("FasterRCNN", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

if __name__ == '__main__':
    main()
