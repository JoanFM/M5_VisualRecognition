from glob import glob
import os
import numpy as np
import cv2

from detectron2.structures import BoxMode

CATEGORIES = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
}
MIT_DATA_DIR = '/home/mcv/datasets/MIT_split/'
KITTI_DATA_DIR = '/home/mcv/datasets/KITTI/'
KITTI_TRAIN_IMG = KITTI_DATA_DIR+'data_object_image_2/training/image_2'
KITTI_TRAIN_LABEL = KITTI_DATA_DIR+'training/label_2'

"""
Labels of KITTI -- Meaning
There are no labels for test images 
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

Example: Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
"""
class Inference_Dataloader():
    def __init__(self, data_dir):
        self.train_paths = glob(data_dir+os.sep+'train/*/*.jpg')
        self.test_paths = glob(data_dir+os.sep+'test/*/*.jpg')

    def load_data(self):
        return {
            'train': self.train_paths,
            'test': self.test_paths
        }

class KITTI_Dataloader():
    def __init__(self, split_perc=0.8):
        if not os.path.isdir(KITTI_TRAIN_IMG):
            raise Exception('The image directory is not correct.')
        if not os.path.isdir(KITTI_TRAIN_LABEL):
            raise Exception('The labels directory is not correct.')
        self.img_paths = np.array(sorted(glob(KITTI_TRAIN_IMG+os.sep+'*.png')))
        self.label_paths = np.array(sorted(glob(KITTI_TRAIN_LABEL+os.sep+'*.txt')))
        img_indices = np.arange(len(self.img_paths))
        split_point = int(img_indices.shape[0]*split_perc)
        self.train_indices = img_indices[:split_point]
        self.test_indices = img_indices[split_point:]
    
    def get_dicts(self, train_flag=False):
        if train_flag is True:
            data_paths = self.img_paths[self.train_indices]
            label_paths = self.label_paths[self.train_indices]
        else:
            data_paths = self.img_paths[self.test_indices]
            label_paths = self.label_paths[self.test_indices]

        dataset_dicts = []
        for k, (img_path, label_path) in enumerate(zip(data_paths,label_paths)):
            record = {}
            filename = os.path.join(img_path)
            height, width = cv2.imread(filename).shape[:2]
            record["file_name"] = filename
            record["image_id"] = k
            record["height"] = height
            record["width"] = width
            objs = []
            with open(label_path,'r') as file:
                lines = file.readlines()
            for line in lines:
                columns = line.split(' ')
                category = CATEGORIES[columns[0]]
                bbox = [float(columns[4]), float(columns[5]), float(columns[6]), float(columns[7])]
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": category
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts
