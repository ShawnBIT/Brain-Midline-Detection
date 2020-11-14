# -*- coding: utf-8 -*-
# File: metrics.py
# Author: Shen Wang <wangshen@pku.edu.cn>

import math
import torch
import numpy as np
from medpy.metric.binary import hd95, assd
from collections import OrderedDict
from .midline import get_ideal_midline, midline2curves
from .utils_metrics import hd_1


class Metric(object):
    def __init__(self, pixel_spacing=0.5):
        self.res_metric_dict = OrderedDict(LDE=[], MSDE=[], hd95=[], assd=[])
        self.ps = pixel_spacing


    def get_final_dict(self):
        final_dict = OrderedDict(LDE=[], MSDE=[], hd95=[], assd=[])

        for key in final_dict.keys():
            mean = np.round(np.mean(self.res_metric_dict[key]), 2)
            std = np.round(np.std(self.res_metric_dict[key]), 2)
            final_dict[key].append(mean)
            final_dict[key].append(std)

        return final_dict


    def process(self, pred_midline_real_limit, pred_midline_gt_limit, gt_midline):
        self.pred_mid_rl = pred_midline_real_limit
        self.pred_mid_gl = pred_midline_gt_limit
        self.gt_mid = gt_midline
        self.batchsize = self.gt_mid.shape[0]

        for b in range(self.batchsize):
            LDE, MSDE = self.get_LDE_MSDE(b)
            hd95_dist, assd_dist = self.get_hd_assd(b)

            self.res_metric_dict['LDE'].append(LDE)
            self.res_metric_dict['MSDE'].append(MSDE)
            self.res_metric_dict['hd95'].append(hd95_dist)
            self.res_metric_dict['assd'].append(assd_dist)

    def get_LDE_MSDE(self, b):
        pm, gm = self.pred_mid_rl[b], self.gt_mid[b]
        diff_coord = np.abs(midline2curves(pm) - midline2curves(gm))
        LDE = np.nanmean(diff_coord)
        MSDE = np.nanmax(diff_coord)

        return LDE * self.ps, MSDE * self.ps


    def get_hd_assd(self, b):
        hd95_dist = hd95(self.pred_mid_rl[b], self.gt_mid[b])
        assd_dist = assd(self.pred_mid_rl[b], self.gt_mid[b])

        return hd95_dist * self.ps, assd_dist * self.ps



# ====== former metric code ======= #

def compute_assd(pred_label, gt_label):
    assert(pred_label.shape == gt_label.shape)
    assd_info = []

    for ii in range(pred_label.shape[0]):
        if (pred_label[ii].sum() == 0 or gt_label[ii].sum() == 0):
            continue
        assd_dist = assd(pred_label[ii], gt_label[ii])
        assd_info.append(assd_dist)

    return assd_info


def compute_hd(pred_label, gt_label):
    assert(pred_label.shape == gt_label.shape)
    hd_info = []

    for ii in range(pred_label.shape[0]):
        if (pred_label[ii].sum() == 0 or gt_label[ii].sum() == 0):
            continue
        hd_dist = hd(pred_label[ii], gt_label[ii])
        hd_info.append(hd_dist)
      
    return hd_info


def compute_dice(pred_label, gt_label):
    assert(pred_label.shape == gt_label.shape)
    dice_info = []

    for ii in range(pred_label.shape[0]):
        union = (torch.sum(pred_label) + torch.sum(gt_label)).float()
        if union > 0:
            intersect = (torch.sum(pred_label * gt_label)).float()
            dice = 2 * intersect / union
        else:
            dice = 1
        dice_info.append(dice)

    return dice_info    
    





















