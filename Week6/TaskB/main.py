import numpy as np
import cv2
from PIL import Image
from glob import glob
from collections import defaultdict
import pickle
from itertools import groupby



def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

if __name__ == "__main__":
    img_paths = sorted(glob('Week6/masks_1/*.png'))
    for k,path in enumerate(img_paths[:5]):
        img = np.array(Image.open(path))
        mask = (img==1)/1
        rle = binary_mask_to_rle(mask)
        cv2.imwrite('priba.png',mask)
        print(np.unique(img))
        print('\n')