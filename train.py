# -*- coding: utf-8 -*-
# File: train.py
# Author: Shen Wang <wangshen@pku.edu.cn>

import os
import os.path as osp
from os.path import exists, join
import argparse
import logging
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn

from lib.factory import get_model, get_optim, get_config
from lib.utils.utils import save_ckpt, load_pretrained_model, resume, count_param, adjust_learning_rate, estimate_final_end_time


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='ours', required=True, type=str)
    args = parser.parse_args()

    return args


def main(cfg):
    # ===== 0.random seed and set logger ===== #
    logger = cfg._get_train_logger()
    logger.write('# ===== MAIN TRAINING ===== #')
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)

    # ===== 1.load train and val dataset ===== #
    logger.write('==> Data loading...')
    train_dataset = cfg.dataset('train', cfg.data_param, cfg.transform_param, logger)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training_setting_param.batch_size_per_gpu * cfg.devices_num,
                                               shuffle=True, num_workers=cfg.training_setting_param.num_workers, pin_memory=True,
                                               drop_last=True)
    val_dataset = cfg.dataset('val', cfg.data_param, cfg.transform_param, logger)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training_setting_param.batch_size_per_gpu * cfg.devices_num,
                                              shuffle=False, num_workers=cfg.training_setting_param.num_workers, pin_memory=False,
                                              drop_last=True)
    logger.write('==> Data loaded Successfully!')
    
    # ===== 2.load the network ===== #
    logger.write('==> Model loading...')
    time.sleep(2)
    net = get_model(cfg.model_param).to(cfg.device)
    input = torch.rand(cfg.transform_param.input_size)
    params = count_param(net)
    logger.write('%s totoal parameters: %.2fM (%d)'%(cfg.model_param.model_name, params/1e6, params))
    
    # ===== 3.define loss and optimizer ===== #
    #cfg.criterion = get_criterion(cfg.loss_param, cfg.task_name)
    cfg.optimizer = get_optim(net, cfg.training_setting_param)
    
    # ===== 4.optionally load a pretrained model or resume from one specific epoch ===== #
    if cfg.pretrained_model_path is not None:
        logger.write("==> loading pretrained model '{}'".format(cfg.pretrained_model_path))
        net = load_pretrained_model(net, cfg.pretrained_model_path)
    if cfg.resume_ckpt_path is not None:
        if os.path.isfile(cfg.resume_ckpt_path):
            logger.write("==> loading checkpoint '{}'".format(cfg.resume_ckpt_path))
            net, cfg.resume_epoch, resume_metric_best = resume(net, cfg.resume_ckpt_path)
            logger.write("==> loaded checkpoint '{}' (epoch {})".format(cfg.resume_ckpt_path, cfg.resume_epoch))
        else:
            logger.write("==> no checkpoint found at '{}'".format(cfg.resume_ckpt_path))
            os._exit(0)

    # ===== 5.main training process ===== #
    cudnn.benchmark = True
    metric_best = resume_metric_best if cfg.resume_ckpt_path is not None else -1000
    start_epoch = cfg.resume_epoch if cfg.resume_ckpt_path is not None else 0
    logger.set_names(['Epoch', 'Loss', 'Metric_Train', 'Metric_Val', 'Is_Best', 'Epoch_Time', 'End_Time'])

    for epoch in range(start_epoch, cfg.training_setting_param.epochs):
        epoch_start_time = time.time()
        print('==> Epoch: [{0}] Prepare Training... '.format(epoch))
        torch.cuda.empty_cache()
        # train for one epoch
        loss_train, metric_train = cfg.train(cfg, net, train_loader, epoch)
        # evaluate on validation set
        metric_val = cfg.val(cfg, net, val_loader)
        # save best checkpoints
        is_best = metric_val > metric_best
        metric_best = max(metric_val, metric_best)
        save_ckpt(cfg, epoch, net, metric_val, metric_best, is_best)
        # log epoch time and estimate the final end
        epoch_time = (time.time() - epoch_start_time) / 60
        final_end_time = estimate_final_end_time(cfg.training_setting_param.epochs, epoch, epoch_time)
        logger.append([epoch, loss_train, metric_train, metric_val, str(int(is_best)) ,epoch_time, final_end_time])
    # from IPython import embed; embed()


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.model)
    main(config)
