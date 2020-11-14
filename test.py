# -*- coding: utf-8 -*-
# File: test.py
# Author: Shen Wang <wangshen@pku.edu.cn>

import os.path as osp
import time

import torch
from torch import nn
import argparse

from lib.factory import get_model, get_config
from lib.utils.utils import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='ours', required=True, type=str)
    args = parser.parse_args()

    return args


def main(cfg):
    # ===== 0.random seed and set logger ===== #
    logger = cfg._get_test_logger()
    logger.write('# === MAIN TEST === #')
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)

    # ===== 1.load test data ===== #
    logger.write('==> Test Data loading...')
    test_dataset = cfg.dataset('test', cfg.data_param, cfg.transform_param, logger)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, 
                                             num_workers=cfg.training_setting_param.num_workers,pin_memory=False, drop_last=True)
    logger.write('==> Test Data loaded Successfully!')

    # ===== 2.load the network ===== #
    logger.write('==> Model loading...')
    net = get_model(cfg.model_param).to(cfg.device)
    assert osp.exists(cfg.test_setting_param.model_path)
    logger.write("==> loading pretrained model '{}'".format(cfg.test_setting_param.model_path))
    net = load_pretrained_model(net, cfg.test_setting_param.model_path)
    logger.write('==> Model loaded Successfully!')

    # ===== 4.main test process ===== #
    cfg.test(cfg, net, test_loader, logger)
    #from IPython import embed; embed()


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.model)
    main(config)
  
