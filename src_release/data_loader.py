import os
import sys
import random
import logging
import torch
from torch.utils import data
from PIL import Image
import torchvision
import numpy as np

from data_tools import decode_mask2img,encode_img2mask
from part_label_dataset import PartLabelDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_dataloader(image_dirs, batch_size, mode='train', num_workers=0, n_classes=3, dataset_name=''):
    """ Build and return data loader """
    transforms = [
        torchvision.transforms.ToTensor(), # scales to 0-1
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # scale data to -1 to 1
    ]
    transforms = torchvision.transforms.Compose(transforms)

    if dataset_name == 'PartLabel':
        dataset_ = PartLabelDataset(image_dirs[0],transforms,mode,augment=(mode=='train'),
                                              n_classes=n_classes)
    else:
        raise NotImplementedError('Dataset %s not implemented' % dataset_name)
        
    dataloader = data.DataLoader(dataset=dataset_,
                                 batch_size=batch_size,
                                 shuffle=(mode=='train'),
                                 num_workers=num_workers,
                                 drop_last=(mode=='train'))
    return dataloader

            
if __name__ == "__main__":
    name2class = {'bg':0,'face':1,'hair':2}
    eval_250_path = os.path.join(os.environ['TMPDIR'],'eval-250/part_label_test_250p')
    eval_dataloader = get_dataloader( (eval_250_path,),
                                  batch_size=64,
                                  mode='eval', num_workers=0,
                                  n_classes=3,
                                  dataset_name='PartLabel')
    batch = iter(eval_dataloader).next()