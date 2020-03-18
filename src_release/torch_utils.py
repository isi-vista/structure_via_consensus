import numpy as np
import torch
import torchvision
from data_tools import decode_mask2img,encode_img2mask
import logging
from torch.nn.functional import upsample
import torch.nn as nn
from conf_matrix import RunningConfusionMatrix
from tqdm import tqdm as tqdm
from skimage import measure
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

'''
Loading of model and optmizers and others train related items
'''
def torch_load_weights(model, optimizer, weight_file, model_only=True):
    weights = torch.load(weight_file)
    model.load_state_dict(weights['state_dict'], strict=True)
    iteration = None
    if 'iteration' in weights:
        iteration = weights['iteration']
    scheduler = None
    if not model_only:
        optimizer.load_state_dict(weights['optimizer'])
        if 'scheduler' in weights:
            scheduler = weights['scheduler']
    return iteration, scheduler

'''
Visualization function for tensorboard and notebook
'''
def tf_viz_img(mask_tmp,i,pred=True):
    if pred:
        mask_tmp = torch.argmax(mask_tmp[i], dim=0).numpy().copy()
    else:
        mask_tmp = mask_tmp[i].numpy().copy()
    mask_tmp = decode_mask2img(mask_tmp)
    mask_tmp = np.transpose(mask_tmp, (2,0,1))
    mask_tmp = mask_tmp / 255.0
    return mask_tmp

'''L1 CC error for two segmentations'''
def error_l1_cc(gt,seg):
    _,num_cc_gt = measure.label(gt,return_num=True)
    _,num_cc_seg = measure.label(seg,return_num=True)
    return abs(num_cc_gt-num_cc_seg)

'''Getting the CC for a segmentation'''
def get_cc(gt):
    _,num_cc_gt = measure.label(gt,return_num=True)
    return num_cc_gt

'''
Connected components for a batch
'''
def get_cc_tensor(tensor_masks):
    if not(isinstance(tensor_masks,np.ndarray)):
        masks = tensor_masks.cpu().numpy()
    else:
        masks = tensor_masks
    ccs = []
    for b in range(masks.shape[0]):
        ccs.append(get_cc(masks[b,...]))
    return np.array(ccs).sum()

'''
Erros over the Connected components for a batch
'''
def err_cc_img(list_gt,list_pred):
    errs = []
    for b in range(list_gt.shape[0]):
        mask_gt = list_gt[b,...]
        mask_pred = list_pred[b,...]
        errs.append(error_l1_cc(mask_gt,mask_pred))
    return np.array(errs)

'''
Updating the confusion matrix batch over batch. 
- Pixel and Super-pixel cases.
- 128p and 250p cases
'''
def update_conf_matrix(conf_mat,list_gt,list_pred,superpixels_gt=None,is250=False):
    if is250:
        list_pred = upsample(list_pred.argmax(dim=1).unsqueeze(1).float(),size=(250,250),mode='nearest').squeeze() 
    else:
        list_pred = list_pred.argmax(dim=1)
    ## we are in the classic pixel regime
    if superpixels_gt is None:
        conf_mat.update_matrix(list_gt.flatten().cpu().numpy(),
                               list_pred.flatten().cpu().numpy())
        errs = err_cc_img(list_gt.cpu().numpy(),list_pred.cpu().numpy())
        conf_mat.update_cc(get_cc_tensor(list_gt),get_cc_tensor(list_pred),errs)
    else:
        #we need to address super pixels. only the sp accuracy is needed in the end
        pred = list_pred.cpu()
        gt = list_gt.cpu()
        gt_sp = superpixels_gt.cpu()
        pred = pred.type(torch.int32)
        ##Loop over batch
        for b in range(pred.shape[0]):
            pred_b = pred[b]
            gt_b = gt[b]
            gt_sp_b = gt_sp[b]
            sp_ids = torch.unique(gt_sp_b)
            sp_all_gts = np.array([])
            sp_all_preds = np.array([])            
            ##Loop over superpixels
            for sp in sp_ids:
                idx_sp = (gt_sp_b == sp)
                ## getting one label
                #(in theory we could just retreive the 1st element but
                #if sp are noisy better apply bincount)
                sp_label = torch.bincount(gt_b[idx_sp]).argmax().type(torch.int32)
                pred_sp_all_labels = pred_b[idx_sp]
                pred_sp_label = torch.bincount(pred_sp_all_labels).argmax()
                ## accumulating all the sp ids and prediction for this image
                sp_all_gts = np.append(sp_all_gts,sp_label.item())
                sp_all_preds = np.append(sp_all_preds,pred_sp_label.item())                
            ## adding all superpixel gt and prediced label to the conf matrix for THIS image
            ## and keep adding for all the images
            conf_mat.update_matrix(sp_all_gts,sp_all_preds)
            
'''
Main Evaluation function
'''
def evaluation(model,validation_dataloader,name2class,is250=False,
                string_label='validation',n_classes=3,super_px=False):
    logger.info('-------------------'+string_label+'------------------')
    # put the model in eval mode to freeze the bn and dropouts
    model.eval()
    box = None
    # Running conf matrxi for pixels
    conf_mat = RunningConfusionMatrix(range(0,n_classes))
    conf_mat_sp = None
    if super_px:
        # Running conf matrix for superpixels
        conf_mat_sp = RunningConfusionMatrix(range(0,n_classes))
    for ibv, batch_val in tqdm(enumerate(validation_dataloader),
                               desc=string_label,
                               total=len(validation_dataloader)):
        img, mask_class, mask_sp = batch_val
        with torch.no_grad():
            img = img.cuda()
            mask_class = mask_class.cuda()
            mask_cc = mask_class.clone().cuda()
            # Get predictions
            pred_mask, conf_mask = model(img)
            # Computing evals
            update_conf_matrix(conf_mat,mask_class,pred_mask,is250=is250)
            if super_px:
                update_conf_matrix(conf_mat_sp,mask_class,pred_mask,is250=is250,superpixels_gt=mask_sp)            

    ### Evaluation
    mean_IoU, IoUs = conf_mat.IoU()
    pix_acc = conf_mat.pixel_accuracy()
    avg_recall, recalls = conf_mat.recall()
    avg_precision, precisions = conf_mat.precision()
    avg_fmeas, f_measure = conf_mat.f_measure()
    err_l1_cc_global,err_l1_cc = conf_mat.err_l1_cc()
    if super_px:    
        spix_acc = conf_mat_sp.pixel_accuracy()
    
    ## Logging
    results = {}
    results['pix_acc'] = pix_acc
    if super_px: 
        results['spix_acc'] = spix_acc
        logger.info(string_label+' global_pix_acc_sp:%f', spix_acc)     
    logger.info(string_label+' global_pix_acc:%f', pix_acc) 
    
    logger.info(string_label+' avg_recall:%f', avg_recall)
    
    for k,v in name2class.items():
        logger.info(string_label+'----')
        results[f'{k}_recall'] = recalls[v]
        results[f'{k}_precision'] = precisions[v]
        results[f'{k}_IoU'] = IoUs[v]
        results[f'{k}_f_measure'] = f_measure[v]  
        logger.info(string_label+' %s_recall:%f'% (k,recalls[v]))
        logger.info(string_label+' %s_precision:%f'% (k,precisions[v]))
        logger.info(string_label+' %s_iou:%f'% (k,IoUs[v]))
        logger.info(string_label+'----')
        logger.info(string_label+' %s_f_measure:%f'%(k,f_measure[v]))
    logger.info(string_label+'----')
    logger.info(string_label+' mean_iou:%f', mean_IoU)
    results['mean_iou'] = mean_IoU
    logger.info(string_label+' err_l1_cc:%f', err_l1_cc)  
    results['err_l1_cc'] = err_l1_cc
    logger.info('-----------------'+string_label+' end.-----------------')
    return conf_mat, conf_mat_sp, results

'''
Main Visualization function
'''
def viz_notebook(model,eval_dataloader,device,ibv_stop=-1):
    import matplotlib.pyplot as plt
    unorm = torchvision.transforms.Compose([ torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2))])
    #batch_val = iter(eval_dataloader).next()
    model.eval()
    with torch.no_grad():
        for ibv, batch_val in tqdm(enumerate(eval_dataloader),
                               desc='viz'):
            img, mask_gt, mask_sp = batch_val
            pred_mask, _ = model(img.to(device))
            pred_mask = pred_mask.cpu()
            mask_gt = mask_gt.cpu().data
            for b in range(pred_mask.shape[0]):
                pred_tmp = tf_viz_img(pred_mask,b,pred=True)
                mask_gt_tmp = tf_viz_img(mask_gt,b,pred=False)
                pred_tmp = np.transpose(pred_tmp, (1,2,0))
                mask_gt_tmp = np.transpose(mask_gt_tmp, (1,2,0))
                ##plotting
                fig = plt.figure()
                plt.subplot(1,3,1)
                plt.title(f'Image {img[b].shape[2]}')
                plt.imshow(np.transpose(unorm(img[b]), (1,2,0)))
                plt.axis('off')
                plt.subplot(1,3,2)
                plt.title(f'Prediction {pred_tmp.shape[0]}')
                plt.imshow(pred_tmp)
                plt.axis('off')
                plt.subplot(1,3,3)
                plt.title(f'Ground-Truth {mask_gt_tmp.shape[0]}')
                plt.imshow(mask_gt_tmp)
                plt.axis('off')                
                plt.show()
                plt.close(fig)
            if ibv_stop == ibv:
                break
                
                
'''
Plot confusion matrix 
'''

def plot_confusion_matrix(cm,name2class):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm.conf_mat/cm.conf_mat.sum(axis=1),
                                    display_labels=[k for k,v in name2class.items()])
    disp.plot(include_values=None,
                    cmap='jet', ax=None, xticks_rotation=None,
                    values_format=None)