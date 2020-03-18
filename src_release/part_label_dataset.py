import os
import sys
import random
import logging
import torch
from torch.utils import data
from PIL import Image
import torchvision
import numpy as np

import aug2D
from data_tools import decode_mask2img

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class PartLabelDataset(data.Dataset):
    """ Dataset class for PartLabel Dataset"""
    def __init__(self, image_dir, transforms, split, augment=False, n_classes=3, model_input=128):
        self.image_dir = image_dir
        self.transforms = transforms
        self.augment = augment
        self.model_input = model_input
        self.is_250p = False
        if 'part_label_test_250p' in image_dir:
            logger.warn('We are processing part labels for 250p case!')
            self.is_250p = True

        self.dataset_2_files = dict()
        self.files = list()
        file_count  = 0
        logger.info("reading the image files...")
        for dirs, subdirs, files in os.walk(self.image_dir):
            for fn in files:
                if '_image_final.png' in fn:
                    image_id = fn.split('_')[-3]
                    subject_id = '_'.join(fn.split('_')[:-3])
                    if (subject_id, image_id) not in self.dataset_2_files:
                        self.files.append({
                            'mask': os.path.join(dirs, fn.replace('_image_final.png', '_cc_occ_labels.png')),
                            'sp_cc': os.path.join(dirs, fn.replace('_image_final.png', '_sp_labels.png')),
                            'image': os.path.join(dirs, fn),
                            'filename' : fn,
                        })
                        self.dataset_2_files[(subject_id, image_id)] = file_count
                        file_count += 1

        self.files = sorted(self.files, key=lambda x: x['image'])
        logger.info("finished initializing the dataloader %d files." % len(self.files))
  
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            ## Getting the data
            if not self.is_250p:
                occl_img = np.array(Image.open(self.files[index]['image']))
            else:
                pilimg = Image.open(self.files[index]['image'])
                pilimg.thumbnail((self.model_input,self.model_input))
                occl_img = np.array(pilimg)
            mask_img = np.array(Image.open(self.files[index]['mask']))
            sp_img = np.array(Image.open(self.files[index]['sp_cc']))            
            mask_img = mask_img.astype(np.uint8)
            if self.augment:
                occl_img, mask_img = aug2D.augment(occl_img,mask_img,
                                                   aug_type='flip',
                                                   patch_size=occl_img.shape[0]
                )
            return self.transforms(occl_img), mask_img, sp_img
            
        except Exception as e:
            logger.error('Error(%s) in dataloader. Continuing to the next index..', e)
            return self.__getitem__((index + 1) % self.__len__())

    ## Method useful to see if the decoding/encoding of occlusion category is working
    def debug(self):
        import os
        for index in range(20):
            occl_img = Image.open(self.files[index]['image'])
            mask_img_loaded = np.array(Image.open(self.files[index]['mask']))
            mask_img = decode_mask2img(mask_class)
            tmpdir = os.environ['TMPDIR']
            result1 = Image.fromarray(mask_img.astype(np.uint8))
            result1.save(tmpdir+'/rbg_mask_%s.png'% str(index).zfill(2))
