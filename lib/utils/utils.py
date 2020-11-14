# -*- coding: utf-8 -*-
# File: utils.py
# Author: Shen Wang <wangshen@pku.edu.cn>

import os
import os.path as osp

import numpy as np
import json
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NpEncoder, self).default(obj)

def ensure_dir(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)


def load_pretrained_model(model,model_path):
    checkpoint = torch.load(model_path)
    model_dict = checkpoint['state_dict']
    model_epoch = checkpoint['epoch']
    print('*** Best model epoch is: %d ***'%model_epoch)
    pretrained_dict = {k: v for k, v in model_dict.items() if k in model.state_dict()}
    model.load_state_dict(pretrained_dict)
    return model


def resume(model, resume_path):
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch']
    best_dice = checkpoint['dice_best']
    model.load_state_dict(checkpoint['state_dict'])
    return model,start_epoch,best_dice


def adjust_learning_rate(cfg, ts_param, epoch, curr_iter, total_iter):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if ts_param.lr_mode == 'step':
        lr = ts_param.lr * (0.1 ** (epoch // ts_param.lr_step))
    elif ts_param.lr_mode == 'poly':
        lr = ts_param.lr * (1 - float(curr_iter) / total_iter) ** ts_param.poly_power
    else:
        raise ValueError('Unknown lr mode {}'.format(ts_param.lr_mode))

    for param_group in cfg.optimizer.param_groups:
        param_group['lr'] = lr

    return lr 


def bad_case_analysis(dice_avg,dice_list):
    for i,dice in enumerate(dice_list):
        if dice < dice_avg and dice >= 0.5*dice_avg:
            print('Bad Case : #{} dice:{}'.format(i,dice))
        elif dice < 0.5*dice_avg:
            print('Very Bad Case : #{} dice:{}'.format(i,dice))


def get_flops_params(model, input):
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")

    return flops, params


def save_ckpt(cfg, epoch, model, dice_val, dice_best, is_best):
    ckpt_dir = cfg.model_dir
    ensure_dir(ckpt_dir)
    ckpt_latest = osp.join(ckpt_dir, 'ckpt_latest.pth.tar')
    state = { 
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'dice_val':dice_val,
        'dice_best': dice_best
        }
    torch.save(state, ckpt_latest)
               
    if (epoch+1) % cfg.save_ckpt_freq == 0:
        ckpt_epoch = osp.join(ckpt_dir,  'ckpt_{}.pth.tar'.format(epoch))
        torch.save(state, ckpt_epoch)
    if is_best:
        ckpt_best = osp.join(ckpt_dir,  'ckpt_best.pth.tar')
        torch.save(state, ckpt_best)
        

def estimate_final_end_time(epochs, epoch, epoch_time):
    epoch_left = epochs - epoch - 1
    final_end_time = (datetime.datetime.now() + datetime.timedelta(minutes=epoch_time*epoch_left)).strftime("%Y-%m-%d %H:%M:%S")

    return final_end_time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def imsave_tensor(img_tensor, src_dir, direction=0, ext='png'):
    for idx in range(img_tensor.shape[direction]):
        if direction == 0:
            cv2.imwrite(src_dir + '/%03d.' % idx + ext, img_tensor[idx, :, :])
        elif direction == 1:
            cv2.imwrite(src_dir + '/%03d.' % idx + ext, img_tensor[:, idx, :])
        elif direction == 2:
            cv2.imwrite(src_dir + '/%03d.' % idx + ext, img_tensor[:, :, idx])


def get_params(model, key):
    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                yield m[1].weight
    if key == '1y':
        for m in model.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                if m[1].weight is not None:
                    yield m[1].weight
    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                if m[1].bias is not None:
                    yield m[1].bias




