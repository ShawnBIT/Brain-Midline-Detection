import numpy as np
from skimage import measure
from medpy.metric.binary import hd
import torch
#import SimpleITK as sitk

def binary_label(pred, gt, foreground):
    pred_label = pred.copy()
    gt_label = gt.copy()
    if(foreground == -1):
        pred_label[pred_label > 0] = 1
        gt_label[gt_label > 0] = 1
    else:
        pred_label[pred_label != foreground] = 0
        gt_label[gt_label != foreground] = 0
        pred_label[pred_label == foreground] = 1
        gt_label[gt_label == foreground] = 1
    return pred_label, gt_label

# def eval_seg(pred, gt, n_class, keys = ['dice', 'region']):
#     res = []
#     for fg_label in range(-1, n_class):
#         if(fg_label == 0):
#             continue        
#         res.append(eval_per_seg(pred_label, gt_label, n_class))
#     return res

#         if(fg_label != -1 and ):
#             dice_array[step, fg_label] = dscore
#             flag_array[step, fg_label] = 1
#         if(fg_label != 0):
#             precision_by_area = 1.0*n_correct/n_predict if n_predict != 0 else 0
#             recall_by_area = 1.0*n_recall/n_gt if n_gt != 0 else 0
#             res_string = '|{:.2f}({:02d}/{:02d})'.format(precision_by_area, n_correct, n_predict)
#             res_string += ',{:.2f}({:02d}/{:02d})'.format(recall_by_area, n_recall, n_gt)
#             res_string += ',{:.2f}|'.format(dscore)
#             res_line.append(res_string)
#             if(fg_label == -1):
#                 area_index = 0
#             else:
#                 area_index = 4*fg_label
#             area_array[step, area_index] = n_correct
#             area_array[step, area_index+1] = n_predict
#             area_array[step, area_index+2] = n_recall
#             area_array[step, area_index+3] = n_gt

def eval_summary(res_all, n_class, keys = ['dice', 'region']):
    res_summary = {}
    for k in keys:
        res_summary[k] = {}
        for fg_label in range(-1, n_class):
            if(fg_label == 0):
                continue
            if(k == 'dice'):
                res_summary['dice'][fg_label] = np.zeros((2))
            if(k == 'region'):
                res_summary['region'][fg_label] = np.zeros((4), dtype = np.int)

    for res in res_all:
        for fg_label in range(-1, n_class):
            if(fg_label == 0):
                continue            
            if('dice' in keys):
                dscore = res['dice'][fg_label]
                if(dscore != -1):
                    res_summary['dice'][fg_label][0] += dscore
                    res_summary['dice'][fg_label][1] += 1
            if('region' in keys):
                n_correct, n_predict, n_recall, n_gt = res['region'][fg_label]        
                res_summary['region'][fg_label] += np.array([n_correct, n_predict, n_recall, n_gt])

    res_out, dscore_ave = eval_to_line(0,res_summary,n_class,keys,mode='all')

    return res_out, dscore_ave

def eval_to_line(step, res, n_class, keys = ['dice', 'region'], mode = 'per'):
    res_out = [step] if mode == 'per' else ['Overall']
    dscore_all , dscount_all = 0.0 , 0
    Dice_list = []
    for fg_label in range(-1, n_class):
        if(fg_label == 0):
            continue
        res_string = ''
        if('region' in keys):
            n_correct, n_predict, n_recall, n_gt = res['region'][fg_label]        
            precision_by_area = 1.0*n_correct/n_predict if n_predict != 0 else 0
            recall_by_area = 1.0*n_recall/n_gt if n_gt != 0 else 0
            res_string += '|{:.2f}({:02d}/{:02d})'.format(precision_by_area, n_correct, n_predict)
            res_string += ',{:.2f}({:02d}/{:02d})'.format(recall_by_area, n_recall, n_gt)
        if mode == 'per':
            if('dice' in keys):
                dscore = res['dice'][fg_label]
                res_string += ',{:.2f}|'.format(dscore)
        else:
            if('dice' in keys):
            #if('dice' in keys and fg_label != -1):
                dscore, dscount = res['dice'][fg_label]
                dscore_all += dscore
                dscount_all += dscount
                dice_i  = dscore/dscount if dscount != 0 else -1
                res_string += ',{:.2f}|'.format(dscore/dscount if dscount != 0 else -1)
                Dice_list.append(dice_i)
        res_out.append(res_string)

    if mode == 'per':
        return res_out
    else:
        dscore_ave = dscore_all/dscount_all if dscount_all != 0 else 0.0
        Dice_list.insert(0,dscore_ave)
        return res_out, Dice_list

def eval_seg(pred, gt, n_class, keys = ['dice', 'region']):
    res = {}
    for k in keys:
        res[k] = {}
    for fg_label in range(-1, n_class):
        if(fg_label == 0):
            continue
        ''' transform the pred and label range into {0, 1} '''
        pred_label, gt_label = binary_label(pred, gt, fg_label)
        if('region' in keys):
            n_correct, n_predict, n_recall, n_gt = compute_region(pred_label, gt_label)
            res['region'][fg_label] = (n_correct, n_predict, n_recall, n_gt)
        if('dice' in keys):
            dscore = compute_dice_score(pred_label, gt_label, mode='cpu') if (gt_label == 1).sum() > 0 else -1
            res['dice'][fg_label] = dscore
    return res


def eval_health(predict, thres = 0):
    connect_pred = measure.label(predict)
    label = np.unique(connect_pred)
    n_predict = 0
    voxel_sum = 0
    for ii in range(1, label.shape[0]):
        if (connect_pred == ii).sum()>thres:
            n_predict += 1
            voxel_sum += (connect_pred == ii).sum()
    return n_predict, voxel_sum

def compute_region(pred_label, gt_label, foreground = 1):
    """rec, prec, ap = eval_seg(pred,
                                gt,
                                foreground)

    Evaluate segmentation according to area recall, precision.

    pred: model prediction        
    gt: ground-truth label         
    foreground: foreground label to evaluate, -1 means using all non-zero labels
    """
    assert(pred_label.shape == gt_label.shape)
    # calculate connect componet for pred label
    connect_pred = measure.label(pred_label)
    connect_label = np.unique(connect_pred)
    n_correct = 0
    n_predict = 0
    thres = 50
    for ii in range(1, connect_label.shape[0]):
        if (connect_pred == ii).sum() > thres:
            n_predict += 1
        if ((connect_pred == ii)*gt_label).sum() > 0 and (connect_pred == ii).sum()>thres:
            n_correct += 1
    # calculate connect componet for ground-truth label
    connect_gt = measure.label(gt_label)
    connect_label = np.unique(connect_gt)
    n_gt = connect_label.shape[0]-1
    n_recall = 0
    for ii in range(1, connect_label.shape[0]):
        if((connect_gt == ii)*pred_label).sum() > thres:
            n_recall += 1
    return n_correct, n_predict, n_recall, n_gt

def compute_pos_acc(pred_label, gt_label, foreground = 1):
    assert(pred_label.shape == gt_label.shape)
    overlap = ((pred_label == foreground)*(gt_label == foreground)).sum()
    pos_acc = 1.0*overlap/(gt_label == foreground).sum()
    return pos_acc

def compute_jaccard(pred_label, gt_label, foreground = 1):
    assert(pred_label.shape == gt_label.shape)
    overlap = ((pred_label == foreground)*(gt_label == foreground)).sum()
    union = ((pred_label == foreground).sum() + (gt_label == foreground).sum()) - overlap
    if overlap > 0 and union > 0 :
        return 2.0 * overlap / union
    else:
        return 0.0

# def compute_dice_score(pred_label, gt_label, foreground = 1):
#     assert(pred_label.shape == gt_label.shape)
#     overlap = ((pred_label == foreground)*(gt_label == foreground)).sum()
#     union = ((pred_label == foreground).sum() + (gt_label == foreground).sum())
#     if overlap > 0 and union > 0 :
#         return 2.0 * overlap / union
#     else:
#         return 0.0


# midline
def compute_hausdorff(pred_label, gt_label, mode='max'):
    assert(pred_label.shape == gt_label.shape)
    hdf_dist = 0.0
    n = 0
    max_slice = 0
    hd_list = []
    #hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    for ii in range(pred_label.shape[0]):
        if(pred_label[ii].sum() == 0 or gt_label[ii].sum() == 0):
            hd_list.append(0)
            continue
        #print(ii,hd(pred_label[ii], gt_label[ii]))
        if mode == 'max':
            # hausdorff_distance_filter.Execute(sitk.GetImageFromArray(pred_label[ii]), sitk.GetImageFromArray(gt_label[ii]))
            # dist = hausdorff_distance_filter.GetHausdorffDistance()
            # hdf_dist = max(hdf_dist, dist)
            dist_i = hd(pred_label[ii], gt_label[ii])
            if dist_i > hdf_dist:
                hdf_dist = dist_i
                max_slice = ii
        elif mode == 'avg':
            # hausdorff_distance_filter.Execute(sitk.GetImageFromArray(pred_label[ii]), sitk.GetImageFromArray(gt_label[ii]))
            # dist = hausdorff_distance_filter.GetHausdorffDistance()
            # hdf_dist += dist
            hdf_dist += hd(pred_label[ii], gt_label[ii])
            n += 1
        elif mode == 'every':
            dist_i = hd(pred_label[ii], gt_label[ii])
            hd_list.append(dist_i)

    if mode == 'max':
        return hdf_dist, max_slice
    elif mode == 'avg':
        final_dist = hdf_dist/n if n != 0 else hdf_dist
        return final_dist
    else:
        return hd_list

    

def compute_connect_area(pred_label):
    n_slice = 0
    n_connect = 0

    for ii in range(pred_label.shape[0]):
        if(pred_label[ii].sum() > 0):
            connect_pred = measure.label(pred_label[ii])
            label = np.unique(connect_pred)
            if len(label) > 2 : 
                n_slice += 1
                n_connect += len(label) - 2

    return n_slice, n_connect


def MSE(pred,gt):
    return 1.0*((pred - gt)**2).sum()/gt.sum()

def compute_mse(pred_label, target):
    MSE_SUM = 0
    N = 0
    for ii in range(pred_label.shape[0]):
        if pred_label[ii].sum() > 0 and target[ii].sum() > 0:
            MSE_SUM += MSE(pred_label[[ii]], target[ii])
            N += 1
    if N > 0:            
        return MSE_SUM*1.0/N
    else:
        return 0



def compute_dice_score(predict, target, n_class=2, foreground=1,mode='gpu'):
    smooth = 0.0
    Dice = 0.0
    count = 0.0

    for i in range(1,n_class):
        if foreground == 'C':
            class_i = i
        else:
            class_i = foreground 
        input_i = (predict == class_i)
        target_i = (target == class_i)
       
        if mode == 'cpu':
            intersect = np.sum(input_i*target_i)
            union = np.sum(input_i) + np.sum(target_i)
        elif mode == 'gpu':
            union = torch.sum(input_i) + torch.sum(target_i)
            union = union.float()
            intersect = torch.sum(input_i*target_i)
            intersect = intersect.float()

        if target_i.sum() > 0:
            count += 1
            dice = (2 * intersect) / union
            Dice += dice 
           
    dice_avg = Dice / count if count != 0 else 0.0
    
    return dice_avg



