# coding=utf-8
# Copyright 2020 USC Information Sciences Institute
# The Software is made available for academic or non-commercial purposes only.
# See the LICENSE file in this repository for more information

"""
Implements the loss enforcing smothness for better structured prediction via consensus used
in the CVPR 2020 paper.
"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import logging 
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class StructureConsensuLossFunction(nn.Module):
    """The structure via consenus loss function.
    The class is initialized with `alpha` and `beta` taht controls the trade-off
    between matching the class label `alpha` and enforcing consenus `beta`.
    The loss takes as input also possibile ways of normailzing each term either normalizing by
    the pixels on a blob (default: idx) or normalizing the blob across all the pixel in the 
    image (HW) option reduce_pixel='all'. Keep in mind that is option 'all' induce a side-effect
    in the training related to inducing a variable temperature based on the size of the blob, 
    that may underperform  so the default option uses 'idx'; the symmetry option for the 
    normalization of the consensus is reduce_pixel_kl='idx' this should be also 
    kept like this to normalize for all the pixels in a blob

    Args:
    @consensus_loss_alpha: hyper-param that ensures we match the label class
    @consensus_loss_beta: novel hyper-param enforcing consensus
    @reduce_pixel: how to normalize 1st term of Eq. (8) 'idx' will normalize by size each blob
    'all' will normalize by HxW where HxW is the size of the image; 'idx' is the one used.
    @reduce_pixel_kl: same as above but how to normalize 2nd term of Eq. (8) related to consensu.
    'idx' will normalize by size each blob
    'all' will normalize by HxW where HxW is the size of the image; 'idx' is the one used.
    """
    def __init__(self,consensus_loss_alpha=10.0,
                 consensus_loss_beta=5.0,
                 reduce_pixel='idx',
                 reduce_pixel_kl='idx'):
        super(StructureConsensuLossFunction,self).__init__()
        self.consensus_loss_alpha=consensus_loss_alpha
        self.consensus_loss_beta=consensus_loss_beta
        self.reduce_pixel=reduce_pixel
        self.reduce_pixel_kl=reduce_pixel_kl
        logger.info(f'\nLoss instanciated with\n'
                    f'alpha: {self.consensus_loss_alpha}\n'
                    f'beta: {self.consensus_loss_beta}\n'
                    f'normalization on 1st (label) term: {self.reduce_pixel}\n'
                    f'normalization on 2nd (consensus) term: {self.reduce_pixel_kl}'
        )
        
    def structure_via_consensus(self,logit,blobs,target):
        '''
        Main function going over the blobs and enforcing consensus

        Args:
        @logit: is the last output for the convolutional decoder *prior* to 
        the softmax normalization
        logit.shape is [batch,channel,size_h,size_w]
        @blobs: this is the masks that contains the connected components to go over each blob and
        enforce
        consensus over them.
        blobs.shape is [batch,1,size_h,size_w]
        @target (y): is the the target label mask
        target.shape is [batch,1,size_h,size_w]
        '''
        total_loss_blobs = 0.
        count = 0.    
        n, c, h, w = logit.size()# batch,channel,size_h,size_w
        blobs = blobs.squeeze(1)
        blobs_cc = torch.unique(blobs,sorted=True,dim=None)
        ## looping over blobs
        for s in blobs_cc:
            idx_blob = blobs == s
            ## Cloning the target labels so we can edit for each blob
            target_blob = target.clone()
            ## Consensus loss in each blob
            loss_pix_avg = self.structure_via_consensus_over_blob(idx_blob,target,logit)
            total_loss_blobs += loss_pix_avg
            count += 1.0
        #normalizing the loss for all blobs
        total_loss_blobs/=count
        return total_loss_blobs

    def structure_via_consensus_over_blob(self,idx_blob,target,logit):
        ## sanity check: all the labels needs to be consistent in each blob
        ## otherwise there is no consensus or bugs in the labels
        assert torch.unique(target[idx_blob]).shape[0] == 1
        ###################################
        ## 1. Computing the average part
        ###################################
        n, c, h, w = logit.size()
        ## building the labels
        ## getting the 1st index in the blob
        ## assumes all pixels in blob have SAME label
        label_id = target[idx_blob][0]
        target_blob_t = label_id.repeat(n).long()
        #Computing probs
        prob = F.softmax(logit,dim=1) #NxCxHxW
        ## building the average
        idx_blob_t = idx_blob.unsqueeze(1).repeat(1,c,1,1)
        idx_blob_c1 = idx_blob.unsqueeze(1)
        ## zeroing the probs not relevant to the blob
        prob_blob = prob*idx_blob_t.float() # NxCxHxW
        support_logit = idx_blob_t.sum(dim=(2,3)) # NxC
        zero_mask = (support_logit==0) #NxC
        ###############################################
        ## Normalization over the pixels for this blob
        ###############################################
        if self.reduce_pixel != 'all':
            # we normalize the sum only on selected index that belongs to the blob
            # in theory with this all the blobs have the same weight
            # respect to each other (i.e. independent of blob size)
            prob_blob_mean, _ = self.custom_div(prob_blob.sum(dim=(2,3)),
                                                support_logit.float() #NxC
            )
        else:
            # we normalize over all the pixel mask (large objects are weighted more in this way)
            prob_blob_mean = prob_blob.sum(dim=(2,3))/(h*w)
        ## Computing NLL (negative loglikehood)
        loss_avg = torch.nn.functional.nll_loss(torch.log(prob_blob_mean),
                                                target_blob_t,
                                                reduction='none')
        invalid_samples = zero_mask.any(dim=1)
        # loss is zero for samples without the class
        loss_avg[invalid_samples] = 0.0
        loss_avg = loss_avg.mean()

        ###################################
        ## 2. Computing the deviation part
        ###################################
        nozero_log_mask = (prob_blob!=0) #select the pixel inside the blob, for which prob!=0
        zero_log_mask = (prob_blob == 0)
        log_prob_blob = prob_blob.clone()
        log_prob_blob[nozero_log_mask] = torch.log(prob_blob[nozero_log_mask]) #NxCxHxW
        prob_blob_mean_target = prob_blob_mean.unsqueeze(dim=(2)).unsqueeze(dim=(3)).repeat(1,1,h,w) #NxCxHxW
        ## the kl loss is sum_i t_i*[log(t_i)-log(x_i)], i \in C
        prob_blob_mean_target[zero_log_mask] = 1.0 #log(1)=0
        log_prob_blob[zero_log_mask] = 0.0
        loss_dev = torch.nn.functional.kl_div(log_prob_blob,
                                              prob_blob_mean_target,
                                              reduction='none') #NxCxHxW
        #sum over classes (kl div) then average over NxHxW
        if self.reduce_pixel_kl == 'all':
            loss_dev = torch.mean(torch.sum(loss_dev,dim=1))
        else:
            loss_dev = torch.sum(loss_dev)/nozero_log_mask.float().sum()
        ## Combining the final loss
        final_loss = self.consensus_loss_alpha*loss_avg + self.consensus_loss_beta*loss_dev
        return final_loss

    def custom_div(self,num,den):
        x = den.clone()
        one_mask = (den!=0) # => one_mask is [0, 1]
        x[one_mask] = num[one_mask]/den[one_mask]
        y=(x)
        zero_mask = (den==0) # => zero_mask is [1, 0]
        y[zero_mask] = 0  # => y is [0, 1]
        return y, zero_mask

    def forward(self,logit,blobs,target):
        return self.structure_via_consensus(logit,blobs,target)
