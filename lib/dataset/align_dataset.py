# -*- coding: utf-8 -*-
# File: dataset.py
# Author: Shen Wang <wangshen@pku.edu.cn>

import copy
import os.path as osp
import sys
sys.path.append('../')

import cv2
import json
import torch
from torch.utils.data import Dataset
import numpy as np 
from PIL import Image

from lib.utils.midline import AlignInfo, get_reg_gt
from lib.utils.img_utils import resize, random_scale, random_rotate, random_mirror, \
                       random_brightness, random_gaussian_blur, normalize


class AlignDataset(Dataset):

    def __init__(self, stage, data, transform, logger):
        self.stage = stage
        assert self.stage in {'train', 'val', 'test'}, "stage must be train or val or test"
        self.data_path_dir = data.data_path_dir
        self.data_file_dir = data.data_file_dir
        self.info_dir = data.info_dir
        self.transform = transform
        self.logger = logger
        self.num_to_train = data.num_to_train
        self.num_to_val = data.num_to_val
        self.num_to_test = data.num_to_test
        self.align_info = AlignInfo()
        self._read_lists()


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, index):
        # ===== fetch data ===== #
        img_path = osp.join(self.data_file_dir, self.image_list[index])
        img, img_align, mid = self._fetch_data(img_path)
    
        # ===== data aug ===== #
        if self.transform.resize:
            img, _ = resize(img, mid, self.transform.resize)
            img_align, _ = resize(img_align, mid, self.transform.resize)
        if self.transform.rotate:
            img, _ = random_rotate(img, mid, self.transform.rotate)
        if self.transform.flip:
            img, _ = random_mirror(img, mid)
        if self.transform.gaussian_kernels:
            img = random_gaussian_blur(img, self.transform.gaussian_kernels)
        if self.transform.brightness:
            img = random_brightness(img, self.transform.brightness)

        img = normalize(img)
        img_align = normalize(img_align)
        
        # ===== torch data format ===== #
        img = torch.FloatTensor(img.transpose(2, 0, 1))
        img_align = torch.FloatTensor(img_align.transpose(2, 0, 1))

        if self.stage == 'test':
            return img, img_align, self.image_list[index]
        else:
            return img, img_align


    def _fetch_data(self, image_path):
        mid_path = image_path.replace('img', 'mid')
        img_align_path = image_path.replace('img', 'img_align')

        img = cv2.imread(image_path)
        img_align = cv2.imread(img_align_path)
        mid = cv2.imread(mid_path, cv2.IMREAD_GRAYSCALE)
        
        return img, img_align, mid
        

    def _read_lists(self):
        # ===== read data list ===== #
        image_path = osp.join(self.data_path_dir, '{}_path.txt'.format(self.stage))
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
