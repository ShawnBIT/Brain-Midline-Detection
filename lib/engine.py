# -*- coding: utf-8 -*-
# File: engine.py
# Author: Shen Wang <wangshen@pku.edu.cn>

import os
import os.path as osp
from os.path import exists, join
import logging
import time
import numpy as np

import cv2
import json
from PIL import Image
import SimpleITK as sitk
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from medpy.metric.binary import hd

from lib.utils.logger import Logger
from lib.utils.utils import AverageMeter, ensure_dir, adjust_learning_rate, MyEncoder
from lib.utils.metrics import compute_hd, Metric, compute_assd
from lib.utils.eval import compute_dice_score, compute_hausdorff, compute_connect_area
from lib.utils.vis import vis_boundary 
from lib.utils.midline import Mid_Argmax, Mid_Threshold, Mid_DP, midline_expand, pred2midline
from lib.utils.img_utils import rotate_successive_3D, shift_3D
from lib.models.loss import get_total_loss


# logger for vis
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)


def train(cfg, model, train_loader, epoch):
    # set the AverageMeter 
    Batch_time = AverageMeter()
    Eval_time = AverageMeter()
    dice = AverageMeter()
    loss_1 = AverageMeter()
    loss_2 = AverageMeter()
    loss_3 = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()
    torch.set_grad_enabled(True)
    # switch to train mode
    model.train()
    start = time.time()
    # loop in dataloader
    for iter, (img, gt_mid, gt_mid_3, gt_x_coords, pos_weight, wce_weight, heatmap) in enumerate(train_loader): 
        curr_iter = iter + 1 + epoch * len(train_loader)
        total_iter = cfg.training_setting_param.epochs * len(train_loader)
        lr = adjust_learning_rate(cfg, cfg.training_setting_param, epoch, curr_iter, total_iter)
        
        input_var = Variable(img).cuda()
        gt_mid_3 = Variable(gt_mid_3).cuda()
        gt_x_coords = Variable(gt_x_coords).cuda()
        pos_weight = Variable(pos_weight).cuda()
        wce_weight = Variable(wce_weight).cuda()
        heatmap = Variable(heatmap).cuda()
        gt_mid_3[gt_mid_3 != 0] = 1
        # forward
        pred = model(input_var)
        # loss
        targets = (gt_x_coords, gt_mid_3, pos_weight, wce_weight, heatmap)
        loss, loss1, loss2, loss3 = get_total_loss(cfg, pred, targets, epoch)
        loss_1.update(loss1.item(), input_var.size(0))
        loss_2.update(loss2.item(), input_var.size(0)) 
        loss_3.update(loss3.item(), input_var.size(0))
        losses.update(loss.item(),  input_var.size(0))
        # mid
        end = time.time()
        # metrics: dice for ven
        Eval_time.update(time.time() - end)
        cfg.optimizer.zero_grad()
        loss.backward()
        cfg.optimizer.step()
        # measure elapsed time
        Batch_time.update(time.time() - start)
        start = time.time()

        if iter % cfg.print_freq == 0:
            logger_vis.info('Epoch-Iter: [{0}/{1}]-[{2}/{3}]\t'
                        'LR: {4:.6f}\t'
                        'Batch_Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Eval_Time {eval_time.val:.3f} ({eval_time.avg:.3f})\t'
                        'Loss1 {loss_1.val:.3f} ({loss_1.avg:.3f})\t'
                        'Loss2 {loss_2.val:.3f} ({loss_2.avg:.3f})\t'
                        'Loss3 {loss_3.val:.3f} ({loss_3.avg:.3f})\t'
                        'Loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                epoch, cfg.training_setting_param.epochs,  iter, len(train_loader), lr, batch_time=Batch_time, eval_time=Eval_time,
                loss_1=loss_1, loss_2=loss_2, loss_3=loss_3, loss=losses))
        
  
    return losses.avg, dice.avg


def val(cfg, model, val_data_loader):
    model.eval()
    torch.set_grad_enabled(False)
    # Indicator to log
    batch_time = AverageMeter()
    HD = []
   
    # The method to predict midline: Left, Right, Max, DP(dynamic programming)
       
    end = time.time() 
    print('# ===== Validation ===== #')    
    for step, (input, target, _, gt_curves, _, _) in enumerate(val_data_loader):
        target[target != 0] = 1
        label = target.numpy().astype('uint8')
        # Variable
        input_var = Variable(input).cuda()
        target_var = target.cuda()
        
        with torch.no_grad():
            # forward       
            pred = model(input_var)
            midline = pred2midline(pred, gt_curves, model_name=cfg.model_param.model_name)[0]
            hd_info = compute_assd(midline, label)
            HD.extend(hd_info)
            batch_time.update(time.time() - end)
            
        end = time.time()
        logger_vis.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(step, len(val_data_loader), batch_time=batch_time))
        #breaK
    hd_mean = np.mean(HD)
   
    return -hd_mean


def test(cfg, model, test_data_loader, logger):
    model.eval()
    torch.set_grad_enabled(False)
    # Indicator to log
    batch_time = AverageMeter()
    shift_list = cfg._get_shift_list()
    metric_class = Metric()
   
    # The method to predict midline: Left, Right, Max, DP(dynamic programming)
       
    end = time.time() 
    print('# ===== TEST ===== #')    
    for step, (input, target, _, gt_curves, ori_img, path) in enumerate(test_data_loader):
        pid = path[0].split('/')[-2]
        target[target != 0] = 1
        gt_midline = target.numpy().astype(np.uint8)
        img = ori_img.numpy().astype(np.uint8)
        # Variable
        input_var = Variable(input).cuda()
        target_var = target.cuda()
        
        with torch.no_grad():
            # forward       
            pred = model(input_var)
            pred_midline_real_limit, pred_midline_gt_limit = pred2midline(pred, gt_curves, model_name=cfg.model_param.model_name)
            metric_class.process(pred_midline_real_limit, pred_midline_gt_limit, gt_midline)
            batch_time.update(time.time() - end)

        # save vis png
        if cfg.test_setting_param.save_vis:
            pid_save_name = pid + '.shift' if pid in shift_list else pid
            save_dir_width_1 = osp.join('{}.{}'.format(cfg.vis_dir, '1'), pid_save_name)
            save_dir_width_3 = osp.join('{}.{}'.format(cfg.vis_dir, '3'), pid_save_name)
            ensure_dir(save_dir_width_1)
            ensure_dir(save_dir_width_3)
            vis_boundary(img, gt_midline, pred_midline_real_limit, save_dir_width_1, is_background=True)
            vis_boundary(img, midline_expand(gt_midline), midline_expand(pred_midline_real_limit), save_dir_width_3, is_background=True)
            print('save vis, finished!')

        end = time.time()
        logger_vis.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(step, len(test_data_loader), batch_time=batch_time))
        
    final_dict = metric_class.get_final_dict()
    assd_mean = final_dict['assd'][0]
    logger.write('\n# ===== final stat ===== #')
    for key, value in final_dict.items():
        logger.write('{}: {}({})'.format(key, value[0], value[1]))
   
    return -assd_mean


