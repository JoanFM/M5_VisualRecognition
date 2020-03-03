from src.tasks import task_inference


def main():
    task_inference("FasterRCNN", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

if __name__ == '__main__':
    main()
