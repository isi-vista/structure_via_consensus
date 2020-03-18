import math
import numpy as np
import random
from skimage import transform as trsf
import sys
import os
from skimage.exposure import  rescale_intensity
from skimage.transform import resize

'''
This random samples scale, 2D rotation and translation
'''
def sample(im,scale=(.95,1),rot=math.pi/4,t=4.):
    ### random sampling
    scale_param = random.uniform(scale[0], scale[1])
    rot_param = random.uniform(-rot, +rot)
    tx_param = random.uniform(-im.shape[1]/t, im.shape[1]/t)
    ty_param = random.uniform(-im.shape[1]/t, im.shape[1]/t)
    return scale_param,rot_param,tx_param,ty_param

'''
This random sampling tx,ty cropping top left corner
'''
def sample_crop(im_shape,crop_size):
    ### random sampleing tx,ty cropping top left corner
    h_off = np.random.randint(0, im_shape[0] - crop_size)
    w_off = np.random.randint(0, im_shape[1] - crop_size)
    return h_off, w_off

'''
Warping function
'''
def warp(im,tform_all,scale_param,order=1,cval=0,mode = 'wrap'):
    aug_im = trsf.warp(im, tform_all, output_shape=(int(im.shape[1]*scale_param),int(im.shape[0]*scale_param)),
                       order=order,preserve_range=True,mode=mode,cval=0)
    aug_im = resize(aug_im, (im.shape[0], im.shape[1]), anti_aliasing=True,order=order)
    return aug_im.astype('uint8')

'''
Cropping function
'''
def crop(im, h_off, w_off, crop_size):
    ##Random flipping the augmented image
    #if np.random.randint(0,2) == 1:
    #    aug_im = cv2.flip(im, 1)
    h_off_max = h_off + crop_size
    w_off_max = w_off + crop_size
    assert h_off >= 0 and h_off < im.shape[0]
    assert w_off >= 0 and w_off < im.shape[1]
    aug_im = im[h_off:h_off_max,w_off:w_off_max]
    return aug_im

'''
Main 2D augmentation function
'''
def augment(im,mask,aug_type='crop',patch_size=128):
    assert im.shape[0] == mask.shape[0]
    assert  im.shape[0] == mask.shape[0]
    aug_im = aug_mask = None
    if aug_type == 'crop':
        #TODO: if we random sample the size then we need to feed a batch with all samples the same sizes [ np.random.randint(64,224) ]
        crop_size = patch_size 
        h_off, w_off = sample_crop(im.shape,crop_size)
        aug_im = crop(im,h_off, w_off, crop_size)
        aug_mask = crop(mask,h_off, w_off, crop_size)        
    elif aug_type == 'sRT':
        scale_param, rot_param, tx_param, ty_param = sample(im)
        
        ### Rotation
        tform_t = trsf.SimilarityTransform(translation=(-im.shape[1]/2.,-im.shape[0]/2.))
        tform_r = trsf.SimilarityTransform(rotation=rot_param)
        tform_tb = trsf.SimilarityTransform(translation=(im.shape[1]/2.,im.shape[0]/2.))

        ## Translation
        tform_tf = trsf.SimilarityTransform(translation=(tx_param,ty_param))

        ## Combing them (R+T)
        tform_all = tform_t + tform_r + tform_tb +  tform_tf

        ## scale
        aug_im = warp(im,tform_all,scale_param,order=1)
        aug_mask = warp(mask,tform_all,scale_param,order=0)
        
        if np.random.rand()<0.5:
            ## we need to copy to avoid this
            ## https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/5
            aug_im = aug_im[:, ::-1,:].copy()
            aug_mask = aug_mask[:, ::-1].copy()
    elif aug_type == 'flip':
        if np.random.rand() < 0.5:
            ## we need to copy to avoid this
            ## https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/5
            aug_im = im[:, ::-1,:].copy()
            aug_mask = mask[:, ::-1].copy()
        else:
            aug_im = im
            aug_mask = mask
    else:
        print('Unkown aug type %s' % aug_type)
    assert (aug_im<0).any() == 0
    assert (aug_mask<0).any() == 0
    return aug_im, aug_mask
