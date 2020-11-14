# -*- coding: utf-8 -*-
# File: common.py
# Author: Shen Wang <wangshen@pku.edu.cn>

import os
import os.path as osp
import time

import torch
from easydict import EasyDict as edict

from lib.utils.logger import Logger
from lib.utils.utils import ensure_dir


class Config:
    # ===== experiment settings ===== #
    project_name = 'miccai.11.1'
    config_name = 'rldn.exp'
   
    devices = "0"
    devices_num = len(devices.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== result root and dir ===== #
    res_root = '/data3/wangshen/results/'
    print('==> The result root:{} and the space left is below.'.format(res_root))
    os.system('df -h /{}'.format(osp.join(res_root.split('/')[1]))) # check mount space

    res_dir = osp.join(res_root, project_name, config_name)
    # ===== train ===== #
    log_dir = osp.join(res_dir, 'log')
    model_dir = osp.join(res_dir, 'model')
    ensure_dir(log_dir)
    ensure_dir(model_dir)
    # ===== test ===== #
    vis_dir = osp.join(res_dir, 'vis')
    npy_dir = osp.join(res_dir, 'npy')


    data_param = edict(
        data_file_dir = '/data3/wangshen/data',
        data_path_dir = '/home/wangshen/data/file_cq/path',
        info_dir = '/home/wangshen/data/file_cq/info',
        fold = 0,
        mid_range_gen = 'rldn',
        heatmap_param = (1, 3), # base_val and delta
        num_to_train = -1,
        num_to_val   = -1,
        num_to_test  = -1
    )
     
    transform_param = edict(
        resize = False,
        rotate = False,
        flip = True, 
        brightness = False, 
        gaussian_kernels = False, 
        input_size = (1, 3, 400, 304)
    )

    model_param = edict(
        model_name = 'rldn',
        n_classes = 2,
        in_channels = 1, 
        feature_scale = 4,
        is_batchnorm = True, 
        up_sample_mode = 'transpose_conv', # transpose_conv, bilinear_1, bilinear_2
        )

    loss_param = edict(
        n_classes = 2,
        seg_type = 'wce', # wce, nce
        with_dice = False, 
        ce_weights = [1, 10], 
        with_connect = False,
        )
    
    training_setting_param = edict(
        batch_size_per_gpu = 16,
        num_workers = 16,
        epochs = 200, 
        lr = 1e-3,
        optim = 'Adam',
        lr_mom = 0.9,
        lr_mode = 'poly',
        lr_step = 15, 
        poly_power = 0.9,
        weight_decay = 1e-4,
        )

    test_setting_param = edict(
        model_path = osp.join(model_dir, 'ckpt_latest.pth.tar'),
        save_vis = True,
        save_npy = True
        )
    
    random_seed = 66
    print_freq = 5
    save_ckpt_freq = 5
    pretrained_model_path = None
    resume_ckpt_path = None

    # ===== task-specific ===== #
    midline_gen_method = 'max'


    def _get_logger(self, logger_name, is_resume):
        logger = Logger(osp.join(self.res_dir, logger_name), title='log', resume=is_resume)
        cur_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        logger.write('# ===== EXPERIMENT TIME: {} ===== #'.format(cur_time), add_extra_empty_line=True)
        logger.write('# ===== CONFIG SETTINGS ===== #')
        for k, v in Config.__dict__.items():
            if not k.startswith('__') and not k.startswith('_'):
                logger.write('{} : {}'.format(k, v))
        logger.write('torch version:{}'.format(torch.__version__), add_extra_empty_line=True)
        
        return logger


    def _get_train_logger(self):
        is_resume = True if self.resume_ckpt_path is not None else False
        logger = self._get_logger(logger_name='train_log.txt', is_resume=is_resume)
        
        return logger
    

    def _get_test_logger(self):
        is_resume = True if osp.exists(osp.join(self.res_dir, 'test_log.txt')) else False
        logger = self._get_logger(logger_name='test_log.txt', is_resume=is_resume)
       
        return logger


    def _get_shift_list(self):
        shift_txt = osp.join(self.data_param.info_dir, 'midline_shift.txt')
        shift_list = [line.strip() for line in open(shift_txt, 'r')]

        return shift_list




config_rldn = Config()


if __name__ == '__main__':
    print(config.project_name)

