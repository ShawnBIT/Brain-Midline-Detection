# -*- coding: utf-8 -*-
# File: dataset.py
# Author: Shen Wang <wangshen@pku.edu.cn>

import copy
import os.path as osp

import cv2
import json
import torch
from torch.utils.data import Dataset
import numpy as np 
from PIL import Image

from lib.utils.img_utils import resize, random_scale, random_rotate, random_mirror, \
                       random_brightness, random_gaussian_blur, normalize
from lib.utils.midline import mid2coord


class SegDataset(Dataset):

    def __init__(self, stage, data, transform, logger):
        self.stage = stage
        assert self.stage in {'train', 'val', 'test'}, "stage must be train or val or test"
        self.data = data
        self.data_path_dir = data.data_path_dir
        self.data_file_dir = data.data_file_dir
        self.info_dir = data.info_dir
        self.fold = data.fold
        self.heatmap_param = data.heatmap_param
        self.transform = transform
        self.logger = logger
        self.num_to_train = data.num_to_train
        self.num_to_val = data.num_to_val
        self.num_to_test = data.num_to_test
        self._read_lists()


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, index):
        '''
        return param 
        gt_1 : mid width 1
        gt_2 : mid width 3
        gt_3 : regress x coords
        '''
        if self.stage == 'train':
            # ===== fetch data ===== #
            img_path = osp.join(self.data_file_dir, self.image_list[index])
            gt_path_1 = img_path.replace('img', 'mid')
            gt_path_2 = img_path.replace('img', 'mid_3')
            img, gt_1, gt_2 = self._fetch_data(img_path, gt_path_1, gt_path_2)
            gt = (gt_1, gt_2)

            # ===== data aug ===== #
            if self.transform.resize:
                img, gt = resize(img, gt, self.transform.resize)
            if self.transform.rotate:
                img, gt = random_rotate(img, gt, self.transform.rotate)
            if self.transform.flip:
                img, gt = random_mirror(img, gt)
            if self.transform.gaussian_kernels:
                img = random_gaussian_blur(img, self.transform.gaussian_kernels)
            if self.transform.brightness:
                img = random_brightness(img, self.transform.brightness)
            img = normalize(img, self.mean, self.std)
            gt_1, gt_2 = gt
            gt_3, pos_weight, wce_weight, heatmap = mid2coord(gt_1, self.heatmap_param, way=self.data.mid_range_gen)
        else:
            # ===== fetch data ===== #
            img_path = osp.join(self.data_file_dir, self.image_list[index])
            gt_path_1 = img_path.replace('img', 'mid')
            gt_path_2 = img_path.replace('img', 'mid_3')
            img, gt_1, gt_2 = self._fetch_data(img_path, gt_path_1, gt_path_2)
            gt_3, pos_weight, wce_weight, heatmap = mid2coord(gt_1, self.heatmap_param, way=self.data.mid_range_gen)
            ori_img = copy.deepcopy(img)
            ori_img = torch.FloatTensor(ori_img)
            
            img = normalize(img, self.mean, self.std)
            
        
        img,  gt_3 = torch.FloatTensor(img), torch.FloatTensor(gt_3)
        gt_1, gt_2 = torch.LongTensor(gt_1), torch.LongTensor(gt_2)
        img = img.unsqueeze(dim=0)

        if self.stage == 'train':
             return img, gt_1, gt_2, gt_3, pos_weight, wce_weight, heatmap
        else:
            return img, gt_1, gt_2, gt_3, ori_img, self.image_list[index]


    def _fetch_data(self, image_path, gt_path_1, gt_path_2):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gt_1 = cv2.imread(gt_path_1, cv2.IMREAD_GRAYSCALE)
        gt_2 = cv2.imread(gt_path_2, cv2.IMREAD_GRAYSCALE)
        
        return img, gt_1, gt_2
        

    def _read_lists(self):
        # ===== read data list ===== #
        image_path = osp.join(self.data_path_dir, '{}_path_{}.txt'.format(self.stage, self.fold))
        self.logger.write('==> image path:{}'.format(image_path))
        assert osp.exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        
        # ===== for debug ===== #
        if self.stage == 'train' and self.num_to_train != -1:
            self.image_list = self.image_list[:self.num_to_train]
        if self.stage == 'val' and self.num_to_val != -1:
            self.image_list = self.image_list[:self.num_to_val]
        if self.stage == 'test' and self.num_to_test != -1:
            self.image_list = self.image_list[:self.num_to_test]

        self.logger.write("==> Stage #{}: load {} images ".format(self.stage, len(self.image_list)))

        # ===== get norm json ===== #
        norm_info = json.load(open(osp.join(self.info_dir, 'norm.json'), 'r'))
        self.mean, self.std = norm_info['mean'], norm_info['std']
