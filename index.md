# Towards Learning Structure via Consensus for Face Segmentation and Parsing
<img src="https://www.isi.edu/images/isi-logo.jpg" width="200"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="http://cvpr2020.thecvf.com/sites/default/files/CVPR_Logo_Horz2_web.jpg" width="200"/>

[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/pdf/1911.00957.pdf)   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-FPLP9uktfW5lXZ0-BgaIoz8nVZZAgoa)

This is the official repo for the Structure via Consensus for Face Segmentation and Parsing (CVPR 2020). 

For method details, please refer to the [paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Masi_Towards_Learning_Structure_via_Consensus_for_Face_Segmentation_and_Parsing_CVPR_2020_paper.pdf) and the [supplementary material](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Masi_Towards_Learning_Structure_CVPR_2020_supplemental.pdf).

<img src='imgs/final_paper.png' />

If you find our method useuful, please cite it as:

```latex
  @inproceedings{masi2020structure,
      title={{T}owards {L}earning {S}tructure via {C}onsensus for {F}ace {S}egmentation and {P}arsing},
      author={Masi, Iacopo and Mathai, Joe and AbdAlmageed, Wael},
      booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2020},
      organization={IEEE/CVF}
  }
```

<img src="imgs/teaser.png" />
<sub>More regular, smooth structure learned; (a) As the training progresses, our method learns more regular, smooth structure which yields a more regular mask when compared to the pixel-wise baseline (sample from the COFW test set); (b) less sparsity is confirmed by visualization of the error in the number of connected components between the predicted and annotated mask.  (c) A higher weight on beta greatly decreases the sparsity of the masks on average.</sub>

# Overview
The gist of the approach is an attempt to induce structure through smoothness in the pixel-wise prediction of a semantic segmentation network trained with a DCNN (Deep Convolutional Neural Networks); the smoothness is enforced for the task of occlusion detection thereby attempting to get occlusion predictions with a better, structured output.
Later on, the method is also used for other similar tasks such as face parsing with 3 classes (skin, background, and facial hair/hair).

_Please, note here the scope is not to get state-of-the-art results on benchmarks yet seeking for a solution on how to enforce structure in the pixel-wise prediction of a DCNN and proving that inducing smoothness can be a feasible and promising direction._



# Dependency
The code is written in python and pytorch. 
A [requirements.txt](requirements.txt) file is avaialble for reference although it contains also unrelated packages.
  
  - Pytorch: 1.1.0
  - Python: 3.7.5
  
Other versions might also work, but not tested.

# New Loss Python package

## Installation

### Typical Install

```
pip install git+https://github.com/isi-vista/structure_via_consensus/loss_package
```

### Typical Usage

```python
from structure_via_consensus import StructureConsensuLossFunction

consensus_loss_alpha=10.0
consensus_loss_beta=5.0
reduce_pixel='idx'
reduce_pixel_kl='idx'

## Define new loss                                                                 
loss_func = StructureConsensuLossFunction(consensus_loss_alpha,                                   
                                     consensus_loss_beta,                                            
                                     reduce_pixel=reduce_pixel,                                                        
                                     reduce_pixel_kl=reduce_pixel_kl                                          
)

for epoch in range(args.nepochs):                                                     
    for ib, batch in enumerate(train_dataloader):                                                        
        img, mask_class, mask_sp = batch                                              
        img = img.cuda()                                                          
        mask_class = mask_class.cuda()                                                  
        mask_cc = mask_class.clone().cuda()                                           
        ## prediction                                                                                                                               
        pred_mask = model(img)                      
        # calculate losses
        loss = loss_func(pred_mask,mask_cc,mask_class)
        # backpropogate
        loss.backward() # compute the gradients and add them
```

Initialize the class:
1. `consensus_loss_alpha`: hyper-param that ensures we match the label class
2. `consensus_loss_beta`: novel hyper-param enforcing consensus
3. `reduce_pixel`: how to normalize 1st term of Eq. (8) `'idx'` will normalize by size each blob.
    `'all'` will normalize by `HxW` where `HxW` is the size of the image;`' idx'` is the one used. Please, note that
       you should NOT use the option `'all'` since from experiments we found out it may underperform because it the 
       normalized probabilities will not sum up to one anymore; if softmax is applied again will bring into issues of 
       introducing a temperature in the softmax function, so stick to `idx` option.
4. `reduce_pixel_kl`: same as above but how to normalize 2nd term of Eq. (8) related to consensus.
    `'idx'` will normalize by size each blob
    `'all'` will normalize by `HxW` where `HxW` is the size of the image;` 'idx'` is the one used.

The input to the loss function is described as:
1. `@pred_mask` **`(Wx)`**: is the last output for the convolutional decoder *prior* to 
the softmax normalization (softmax normalization is done inside the function)
pred_mask.shape is `[batch,channel,size_h,size_w]`
2. `@mask_cc` **`(C)`**: this is the masks that contains the connected components to go over each blob and
enforce consensus over them. This can be same as the label mask or different. Please, refer to Fig. 2 in the paper in the case they are different from the labels
mask_cc.shape is `[batch,1,size_h,size_w]`
3. `@mask_class` **`(y)`**: is the the target label mask as usual in semantic segmentation. Each pixel indicates the index of the class label for that pixel.
mask_class.shape is `[batch,1,size_h,size_w]`

#### Hyper-params on the loss
We shows that the hyper-param `beta` correlates with decreasing the sparsity of the ouput mask on average although in our experimentation when fine-tuning on PartLabel and COFW we found out that increasing `beta` too much may underfit the prediction to the labels in our training time regime. We used a cross-validated ratio of `alpha:beta` as `alpha=10,beta=5` although these values may be dataset dependent.
Also, it is important to notice that **all the metrics (pixel acc., IoU, etc) currently considered in semantic segmentation** do not take sparsity or smoothness into consideration [nor countour measures](http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf).

# Part Label Dataset Demo
One may simply download the repo and play with the provided ipython notebook [eval_model_pt_labels_colab.ipynb](eval_model_pt_labels_colab.ipynb).

Alternatively, one may play with the inference code using [this Google Colab link](https://colab.research.google.com/drive/1-FPLP9uktfW5lXZ0-BgaIoz8nVZZAgoa).

<img src="imgs/pt_1.png" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="imgs/pt_2.png" /> 
# Contact
For any paper related questions, please contact `masi@isi.edu` or open a Github issues above. Please, note that we do not provide software support besides simply replying to a few questions. Thank you for your understanding.

# [License](LICENSE)
The Software is made available for academic or non-commercial purposes only. The license is for a copy of the program for an unlimited term. Individuals requesting a license for commercial use must pay for a commercial license.

    USC Stevens Institute for Innovation 
    University of Southern California 
    1150 S. Olive Street, Suite 2300 
    Los Angeles, CA 90115, USA 
    ATTN: Accounting 
 
DISCLAIMER. USC MAKES NO EXPRESS OR IMPLIED WARRANTIES, EITHER IN FACT OR BY OPERATION OF LAW, BY STATUTE OR OTHERWISE, AND USC SPECIFICALLY AND EXPRESSLY DISCLAIMS ANY EXPRESS OR IMPLIED WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, VALIDITY OF THE SOFTWARE OR ANY OTHER INTELLECTUAL PROPERTY RIGHTS OR NON-INFRINGEMENT OF THE INTELLECTUAL PROPERTY OR OTHER RIGHTS OF ANY THIRD PARTY. SOFTWARE IS MADE AVAILABLE AS-IS. LIMITATION OF LIABILITY. TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT WILL USC BE LIABLE TO ANY USER OF THIS CODE FOR ANY INCIDENTAL, CONSEQUENTIAL, EXEMPLARY OR PUNITIVE DAMAGES OF ANY KIND, LOST GOODWILL, LOST PROFITS, LOST BUSINESS AND/OR ANY INDIRECT ECONOMIC DAMAGES WHATSOEVER, REGARDLESS OF WHETHER SUCH DAMAGES ARISE FROM CLAIMS BASED UPON CONTRACT, NEGLIGENCE, TORT (INCLUDING STRICT LIABILITY OR OTHER LEGAL THEORY), A BREACH OF ANY WARRANTY OR TERM OF THIS AGREEMENT, AND REGARDLESS OF WHETHER USC WAS ADVISED OR HAD REASON TO KNOW OF THE POSSIBILITY OF INCURRING SUCH DAMAGES IN ADVANCE.

For commercial license pricing and annual commercial update and support pricing, please contact:

 
    Nikolaus Traitler USC Stevens Institute for Innovation
    University of Southern California
    1150 S. Olive Street, Suite 2300
    Los Angeles, CA 90115, USA
 
    Tel: +1 213-821-3550
    Fax: +1 213-821-5001
    Email: traitler@usc.edu and cc to: accounting@stevens.usc.edu
