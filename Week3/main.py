from src.tasks import KITTIMOTS_inference_task


def main():
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

    # -- TASK B --

    KITTIMOTS_inference_task("FasterRCNN_X_101", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    KITTIMOTS_inference_task("FasterRCNN_R_101", "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    KITTIMOTS_inference_task("FasterRCNN_R_50_DC5", "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml")
    KITTIMOTS_inference_task("FasterRCNN_R_50_C4", "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
    
    KITTIMOTS_inference_task("RetinaNET_R_50", "COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    KITTIMOTS_inference_task("RetinaNET_R_101", "COCO-Detection/retinanet_R_101_FPN_3x.yaml")


    # -- TASK C --

    # train_task("FasterRCNN_R_101", "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

if __name__ == '__main__':
    main()
