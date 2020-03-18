import os
import sys
import random
import logging
from PIL import Image
import numpy as np


def encode_img2mask(image):
    """
    Function to encode a synthetic image saved as opencv BGR as HxWx3 tensor 
    to a HxWx1 where the last channel represents the occlusion category ID
    """
    image = image.astype(int)
    label_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_rgb_labels()):
        label_mask[np.where(np.all(image == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def decode_mask2img(mask):
    image_mask = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    rgb_labels = get_rgb_labels()
    classes = np.unique(mask)
    for c in classes:
        idx = np.where(mask==c)
        idx_c_mod = c % rgb_labels.shape[0]
        image_mask[idx[0],idx[1],:] = rgb_labels[idx_c_mod]
    return image_mask

def get_rgb_labels(bgr=False,face_segm=False):
    """Load the mapping that associates pascal classes with label colors
    Returns:
    np.ndarray with dimensions (#category, 3)
    """
    if not face_segm:
        labels = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                         [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                         [0, 64, 128]])
    else:
        labels = np.asarray([[0, 0, 0], [255, 255, 255]])
    ## in case images are loaded BGR (with cv2)    
    if bgr:
        return labels
    else:
        return labels[:,::-1]
