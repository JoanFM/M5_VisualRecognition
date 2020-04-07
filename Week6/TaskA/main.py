from src.tasks import task_a


def main():

    # TASK A
    task_a("MaskRCNN_R_50_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", save_name='none2')


    # da = {
    #     'crop': {
    #         'type': 'relative',
    #         'size': [0.9, 0.9]
    #     }
    # }
    # task_a("MaskRCNN_R_50_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", save_name='crop2', da=da)


    # da = {
    #     'flip': {
    #         'prob': 0.5,
    #         'horizontal': True,
    #         'vertical': False
    #     }
    # }
    # task_a("MaskRCNN_R_50_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", save_name='flip2', da=da)


    # da = {
    #     'crop': {
    #         'type': 'relative',
    #         'size': [0.9, 0.9]
    #     },
    #     'flip': {
    #         'prob': 0.5,
    #         'horizontal': True,
    #         'vertical': False
    #     }
    # }
    # task_a("MaskRCNN_R_50_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", save_name='crop_flip2', da=da)
    


if __name__ == '__main__':
    main()