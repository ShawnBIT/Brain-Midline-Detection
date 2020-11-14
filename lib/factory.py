
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')

from .dataset.seg_dataset import SegDataset
from .models.network import Ours_Baseline, Ours_CARNet, MD_Net, NetworkRussia, RLDN
from .models.loss import Criterion
from .utils.utils import get_params
from .engine import train, val, test


def get_model(model_param):
    if model_param.model_name == 'ours_baseline':
        net = Ours_Baseline(n_classes=model_param.n_classes, 
        	       in_channels = model_param.in_channels, 
        	       feature_scale=model_param.feature_scale,
        	       is_batchnorm=model_param.is_batchnorm,
        	       up_sample_mode=model_param.up_sample_mode)
    elif model_param.model_name == 'mdnet':
        net = MD_Net(n_classes=model_param.n_classes, 
                 in_channels = model_param.in_channels, 
                 feature_scale=model_param.feature_scale,
                 is_batchnorm=model_param.is_batchnorm,
                 up_sample_mode=model_param.up_sample_mode)
    elif model_param.model_name == 'ours_carnet':
        net = Ours_CARNet(n_classes=model_param.n_classes, 
                   in_channels = model_param.in_channels, 
                   feature_scale=model_param.feature_scale,
                   is_batchnorm=model_param.is_batchnorm,
                   up_sample_mode=model_param.up_sample_mode,
                   beta=model_param.beta)
    elif model_param.model_name == 'russia':
        net = NetworkRussia()
    elif model_param.model_name == 'rldn':
        net = RLDN(input_size=(400, 304), S=5)
     
    return net


def get_optim(net, training_setting_param):
	assert training_setting_param.optim in ['Adam', 'SGD']
	
	if training_setting_param.optim == 'Adam':
		optimizer = torch.optim.Adam(net.parameters(), lr=training_setting_param.lr,
			betas=(0.9, 0.99), weight_decay=training_setting_param.weight_decay)
	elif training_setting_param.optim == 'SGD':
		optimizer = torch.optim.SGD(net.parameters(), lr=training_setting_param.lr, \
			        momentum=training_setting_param.lr_mom)

	return optimizer


def get_config(model_name):
    if model_name == 'ours':
        from config.config_ours import config_ours
        config_ours.dataset = SegDataset
        config_ours.train = train
        config_ours.val = val
        config_ours.test = test

        return config_ours

    elif model_name == 'russia':
        from config.config_russia import config_russia 
        config_russia.dataset = SegDataset
        config_russia.train = train
        config_russia.val = val
        config_russia.test = test
        return config_russia

    elif model_name == 'rldn':
        from config.config_rldn import config_rldn 
        config_rldn.dataset = SegDataset
        config_rldn.train = train
        config_rldn.val = val
        config_rldn.test = test
        return config_rldn

    elif model_name == 'mdnet':
        from config.config_mdnet import config_mdnet
        config_mdnet.dataset = SegDataset
        config_mdnet.train = train
        config_mdnet.val = val
        config_mdnet.test = test
        return config_mdnet
