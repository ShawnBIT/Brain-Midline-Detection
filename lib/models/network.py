import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from dpipe import layers

from .vgg import decompose_vgg16
from .resnet import BasicBlock
from .init_weights import init_weights
from .layers import ChannelSpatialSELayer, unetUp_nested, unetConv2, unetUp, Fuse, Merge, SoftArgmax, PSPModule, CoordConv2d, SELayer, CoordGuidedAttention, unetConv2_dilation

class MD_Net(nn.Module):

    def __init__(self, in_channels=1, n_classes=2,feature_scale=4, is_deconv=True, is_batchnorm=True, up_sample_mode='transpose_conv'):
        super(MD_Net, self).__init__()
        self.up_sample_mode = up_sample_mode
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(filters[0], filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.coordconv2d_1 = CoordConv2d(self.in_channels, filters[0], 3, 1, 1, with_r = False)
        self.dialted = unetConv2_dilation(filters[4],filters[4],self.is_batchnorm)  

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.up_sample_mode)
        self.up_concat3 = unetUp(filters[3], filters[2], self.up_sample_mode)
        self.up_concat2 = unetUp(filters[2], filters[1], self.up_sample_mode)
        self.up_concat1 = unetUp(filters[1], filters[0], self.up_sample_mode)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], 1, 1)
        self.final_2 = nn.Conv2d(filters[1], 1, 1)
        self.final_3 = nn.Conv2d(filters[2], 1, 1)
        self.final_4 = nn.Conv2d(filters[3], 1, 1)

        self.sigmoid = nn.Sigmoid()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(self.coordconv2d_1(inputs))       # 16*512*512
        maxpool1 = self.maxpool1(conv1)  # 16*256*256
        
        conv2 = self.conv2(maxpool1)     # 32*256*256
        maxpool2 = self.maxpool2(conv2)  # 32*128*128

        conv3 = self.conv3(maxpool2)     # 64*128*128
        maxpool3 = self.maxpool3(conv3)  # 64*64*64

        conv4 = self.conv4(maxpool3)     # 128*64*64
        maxpool4 = self.maxpool4(conv4)  # 128*32*32

        center = self.dialted(self.center(maxpool4))   # 256*32*32
        
        up4 = self.up_concat4(center,conv4)  # 128*64*64
        up3 = self.up_concat3(up4,conv3)     # 64*128*128
        up2 = self.up_concat2(up3,conv2)     # 32*256*256
        up1 = self.up_concat1(up2,conv1)     # 16*512*512

        final_1 = self.final_1(up1).squeeze(dim=1)
        final_2 = self.final_2(up2).squeeze(dim=1)
        final_3 = self.final_3(up3).squeeze(dim=1)
        final_4 = self.final_4(up4).squeeze(dim=1)

        final_1 = self.sigmoid(final_1)
        final_2 = self.sigmoid(final_2)
        final_3 = self.sigmoid(final_3)
        final_4 = self.sigmoid(final_4)

        return final_1,final_2,final_3,final_4


class Ours_Nested(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True, up_sample_mode='transpose_conv'):
        super(Ours_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.up_sample_mode = up_sample_mode
       
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        downsample = partial(nn.MaxPool2d, kernel_size=2)
        self.limits_head = layers.InterpolateToInput(nn.Sequential(
            layers.ResBlock2d(16, 32, kernel_size=3, padding=1),
            downsample(),
            layers.ResBlock2d(32, 32, kernel_size=3, padding=1),

            # 2D feature map to 1D feature map
            nn.AdaptiveMaxPool2d((None, 1)),
            layers.Reshape('0', '1', -1),

            nn.Conv1d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        ), mode='linear', axes=0)

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp_nested(filters[1], filters[0], self.up_sample_mode)
        self.up_concat11 = unetUp_nested(filters[2], filters[1], self.up_sample_mode)
        self.up_concat21 = unetUp_nested(filters[3], filters[2], self.up_sample_mode)
        self.up_concat31 = unetUp_nested(filters[4], filters[3], self.up_sample_mode)

        self.up_concat02 = unetUp_nested(filters[1], filters[0], self.up_sample_mode, 3)
        self.up_concat12 = unetUp_nested(filters[2], filters[1], self.up_sample_mode, 3)
        self.up_concat22 = unetUp_nested(filters[3], filters[2], self.up_sample_mode, 3)

        self.up_concat03 = unetUp_nested(filters[1], filters[0], self.up_sample_mode, 4)
        self.up_concat13 = unetUp_nested(filters[2], filters[1], self.up_sample_mode, 4)
        
        self.up_concat04 = unetUp_nested(filters[1], filters[0], self.up_sample_mode, 5)
        
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], 1, 1)
        self.sigmoid = nn.Sigmoid()
    
        # ===== regression_based_refine ===== #
        self.downsample_a = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32))
        self.res_blocks_a = BasicBlock(inplanes=1, planes=32, stride=2, downsample=self.downsample_a)
        self.downsample_b = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64))
        self.res_blocks_b = BasicBlock(inplanes=32, planes=64, stride=2, downsample=self.downsample_b)
        self.downsample_c = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128))
        self.res_blocks_c = BasicBlock(inplanes=64, planes=128, stride=2, downsample=self.downsample_c)

        self.softargmax = SoftArgmax(beta=10)
        self.res_blocks_mid = nn.Sequential(*[self.res_blocks_a, self.res_blocks_b, self.res_blocks_c])
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.regress_mid = nn.Sequential(
                        nn.Linear(128 * 4 * 4, 600),
                        nn.Linear(600, 400))
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)       # 16*512*512
        maxpool0 = self.maxpool(X_00)    # 16*256*256
        X_10= self.conv10(maxpool0)      # 32*256*256
        maxpool1 = self.maxpool(X_10)    # 32*128*128
        X_20 = self.conv20(maxpool1)     # 64*128*128
        maxpool2 = self.maxpool(X_20)    # 64*64*64
        X_30 = self.conv30(maxpool2)     # 128*64*64
        maxpool3 = self.maxpool(X_30)    # 128*32*32
        X_40 = self.conv40(maxpool3)     # 256*32*32
        # column : 1
        X_01 = self.up_concat01(X_10,X_00)
        X_11 = self.up_concat11(X_20,X_10)
        X_21 = self.up_concat21(X_30,X_20)
        X_31 = self.up_concat31(X_40,X_30)
        # column : 2
        X_02 = self.up_concat02(X_11,X_00,X_01)
        X_12 = self.up_concat12(X_21,X_10,X_11)
        X_22 = self.up_concat22(X_31,X_20,X_21)
        # column : 3
        X_03 = self.up_concat03(X_12,X_00,X_01,X_02)
        X_13 = self.up_concat13(X_22,X_10,X_11,X_12)
        # column : 4
        X_04 = self.up_concat04(X_13,X_00,X_01,X_02,X_03)

        # final layer
        prob_map = self.final(X_04).squeeze(dim=1)
        prob_map = self.sigmoid(prob_map)

        # ===== limits head ===== # 
        limits = self.limits_head(X_00)
    
        # ===== midline head ===== #
        ref_curves = self.res_blocks_mid(prob_map.unsqueeze(dim=1))
        ref_curves = self.avgpool(ref_curves)
        ref_curves = torch.flatten(ref_curves, 1)
        ref_curves = self.regress_mid(ref_curves)
        curves = self.softargmax(prob_map) + ref_curves
        curves = curves.unsqueeze(dim=1)
    
        return prob_map, torch.cat([curves, limits], 1)


class Ours_Baseline(nn.Module):

    def __init__(self, n_classes=2, in_channels=3, feature_scale=4, is_batchnorm=True, up_sample_mode='transpose_conv'):
        super(Ours_Baseline, self).__init__()
        self.up_sample_mode = up_sample_mode
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        downsample = partial(nn.MaxPool2d, kernel_size=2)
        self.limits_head = layers.InterpolateToInput(nn.Sequential(
            layers.ResBlock2d(16, 32, kernel_size=3, padding=1),
            downsample(),
            layers.ResBlock2d(32, 32, kernel_size=3, padding=1),

            # 2D feature map to 1D feature map
            nn.AdaptiveMaxPool2d((None, 1)),
            layers.Reshape('0', '1', -1),

            nn.Conv1d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        ), mode='linear', axes=0)

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.up_sample_mode)
        self.up_concat3 = unetUp(filters[3], filters[2], self.up_sample_mode)
        self.up_concat2 = unetUp(filters[2], filters[1], self.up_sample_mode)
        self.up_concat1 = unetUp(filters[1], filters[0], self.up_sample_mode)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], 1, 1)
        self.sigmoid = nn.Sigmoid()
    
        # ===== regression_based_refine ===== #
        self.downsample_a = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32))
        self.res_blocks_a = BasicBlock(inplanes=1, planes=32, stride=2, downsample=self.downsample_a)
        self.downsample_b = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64))
        self.res_blocks_b = BasicBlock(inplanes=32, planes=64, stride=2, downsample=self.downsample_b)
        self.downsample_c = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128))
        self.res_blocks_c = BasicBlock(inplanes=64, planes=128, stride=2, downsample=self.downsample_c)

        self.softargmax = SoftArgmax(beta=10)
        self.res_blocks_mid = nn.Sequential(*[self.res_blocks_a, self.res_blocks_b, self.res_blocks_c])
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.regress_mid = nn.Sequential(
                        nn.Linear(128 * 4 * 4, 600),
                        nn.Linear(600, 400))
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)      
        maxpool1 = self.maxpool1(conv1)  
        
        conv2 = self.conv2(maxpool1)     
        maxpool2 = self.maxpool2(conv2) 

        conv3 = self.conv3(maxpool2)     
        maxpool3 = self.maxpool3(conv3) 

        conv4 = self.conv4(maxpool3)    
        maxpool4 = self.maxpool4(conv4)  

        center = self.center(maxpool4)  
        up4 = self.up_concat4(center,conv4)  
        up3 = self.up_concat3(up4,conv3)     
        up2 = self.up_concat2(up3,conv2)    
        up1 = self.up_concat1(up2,conv1)   

        prob_map = self.final(up1).squeeze(dim=1)
        prob_map = self.sigmoid(prob_map)

        # ===== limits head ===== # 
        limits = self.limits_head(conv1)
    
        # ===== midline head ===== #
        ref_curves = self.res_blocks_mid(prob_map.unsqueeze(dim=1))
        ref_curves = self.avgpool(ref_curves)
        ref_curves = torch.flatten(ref_curves, 1)
        ref_curves = self.regress_mid(ref_curves)
        curves = self.softargmax(prob_map) + ref_curves
        curves = curves.unsqueeze(dim=1)
    
        return prob_map, torch.cat([curves, limits], 1)

        

class Ours_CARNet(nn.Module):

    def __init__(self, n_classes=2, in_channels=3, feature_scale=4, is_batchnorm=True, up_sample_mode='transpose_conv', beta=10):
        super(Ours_CARNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.beta = beta

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        downsample = partial(nn.MaxPool2d, kernel_size=2)
        self.limits_head = layers.InterpolateToInput(nn.Sequential(
            layers.ResBlock2d(16, 32, kernel_size=3, padding=1),
            downsample(),
            layers.ResBlock2d(32, 32, kernel_size=3, padding=1),

            # 2D feature map to 1D feature map
            nn.AdaptiveMaxPool2d((None, 1)),
            layers.Reshape('0', '1', -1),

            nn.Conv1d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        ), mode='linear', axes=0)

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.up_sample_mode)
        self.up_concat3 = unetUp(filters[3], filters[2], self.up_sample_mode)
        self.up_concat2 = unetUp(filters[2], filters[1], self.up_sample_mode)
        self.up_concat1 = unetUp(filters[1], filters[0], self.up_sample_mode)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], 1, 1)
        self.sigmoid = nn.Sigmoid()

        # ===== refine net ===== #
        self.laternal_1 = nn.Conv2d(filters[0], filters[0], 1)
        self.laternal_2 = nn.Conv2d(filters[1], filters[0], 1)
        self.laternal_3 = nn.Conv2d(filters[2], filters[0], 1)
        self.laternal_4 = nn.Conv2d(filters[3], filters[0], 1)
        self.laternal_5 = nn.Conv2d(filters[4], filters[0], 1)
        
        self.refine_conv_1 = SELayer(filters[0])
        self.refine_conv_2 = nn.Sequential(*[
                             unetConv2(filters[0], filters[0], self.is_batchnorm),\
                             SELayer(filters[0])
                             ])
        self.refine_conv_3 = nn.Sequential(*[
                             unetConv2(filters[0], filters[0], self.is_batchnorm),\
                             unetConv2(filters[0], filters[0], self.is_batchnorm),\
                             SELayer(filters[0])
                             ])
        self.refine_conv_4 = nn.Sequential(*[
                             unetConv2(filters[0], filters[0], self.is_batchnorm),\
                             unetConv2(filters[0], filters[0], self.is_batchnorm),\
                             unetConv2(filters[0], filters[0], self.is_batchnorm),\
                             SELayer(filters[0])
                             ])
        self.refine_conv_5 = nn.Sequential(*[
                             unetConv2(filters[0], filters[0], self.is_batchnorm),\
                             unetConv2(filters[0], filters[0], self.is_batchnorm),\
                             unetConv2(filters[0], filters[0], self.is_batchnorm),\
                             unetConv2(filters[0], filters[0], self.is_batchnorm),\
                             SELayer(filters[0])
                             ])
        
        self.refine_final = unetConv2(filters[0] * 5, filters[0], self.is_batchnorm)
        self.final_2 = nn.Conv2d(filters[0], 1, 1)

    
        # ===== regression_based_refine ===== #
        self.downsample_a = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32))
        self.res_blocks_a = BasicBlock(inplanes=1, planes=32, stride=2, downsample=self.downsample_a)
        self.downsample_b = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64))
        self.res_blocks_b = BasicBlock(inplanes=32, planes=64, stride=2, downsample=self.downsample_b)
        self.downsample_c = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128))
        self.res_blocks_c = BasicBlock(inplanes=64, planes=128, stride=2, downsample=self.downsample_c)

        self.softargmax = SoftArgmax(beta=self.beta)
        self.res_blocks_mid = nn.Sequential(*[self.res_blocks_a, self.res_blocks_b, self.res_blocks_c])
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.regress_mid = nn.Sequential(
                        nn.Linear(128 * 4 * 4, 600),
                        nn.Linear(600, 400))
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)      
        maxpool1 = self.maxpool1(conv1)  
        
        conv2 = self.conv2(maxpool1)     
        maxpool2 = self.maxpool2(conv2) 

        conv3 = self.conv3(maxpool2)     
        maxpool3 = self.maxpool3(conv3) 

        conv4 = self.conv4(maxpool3)    
        maxpool4 = self.maxpool4(conv4)  

        center = self.center(maxpool4)  
        up4 = self.up_concat4(center,conv4)  
        up3 = self.up_concat3(up4,conv3)     
        up2 = self.up_concat2(up3,conv2)    
        up1 = self.up_concat1(up2,conv1)   

        prob_map_1 = self.final_1(up1).squeeze(dim=1)
        prob_map_1 = self.sigmoid(prob_map_1)

        # ===== refine net ===== # 
        up1 = self.refine_conv_1(self.laternal_1(up1))
        up2 = self.refine_conv_2(self.laternal_2(up2))
        up3 = self.refine_conv_3(self.laternal_3(up3))
        up4 = self.refine_conv_4(self.laternal_4(up4))
        up5 = self.refine_conv_5(self.laternal_5(center))

        #up1 = F.interpolate(up1, scale_factor=1,  mode='bilinear', align_corners=True)
        up2 = F.interpolate(up2, scale_factor=2,  mode='bilinear', align_corners=True)
        up3 = F.interpolate(up3, scale_factor=4,  mode='bilinear', align_corners=True)
        up4 = F.interpolate(up4, scale_factor=8,  mode='bilinear', align_corners=True)
        up5 = F.interpolate(up5, scale_factor=16, mode='bilinear', align_corners=True)
    
        refine_ft = torch.cat([up1, up2, up3, up4, up5], dim=1)
        #refine_ft = F.interpolate(refine_ft, scale_factor=4, mode='bilinear', align_corners=True)
        refine_ft = self.refine_final(refine_ft)
        prob_map = self.final_2(refine_ft).squeeze(dim=1)
        prob_map = self.sigmoid(prob_map)

        # ===== limits head ===== # 
        limits = self.limits_head(conv1)
    
        # ===== midline head ===== #
        ref_curves = self.res_blocks_mid(prob_map.unsqueeze(dim=1))
        ref_curves = self.avgpool(ref_curves)
        ref_curves = torch.flatten(ref_curves, 1)
        ref_curves = self.regress_mid(ref_curves)
        curves = self.softargmax(prob_map) + ref_curves
        curves = curves.unsqueeze(dim=1)

        prob_maps = [prob_map_1, prob_map]
    
        return prob_maps, torch.cat([curves, limits], 1)

# ===== Russia Paper ===== #

def merge_branches(left, down):
    # interpolate the `down` branch to the same shape as `left`
    interpolated = F.interpolate(down, size=left.shape[2:], mode='bilinear', align_corners=False)
    return torch.cat([left, interpolated], dim=1)


def expectation(distribution):
    rng = torch.arange(distribution.shape[-1]).to(distribution)

    return (distribution * rng).sum(-1)


STRUCTURE = [
    [[32, 32], [64, 32, 32]],
    [[32, 64, 64], [128, 64, 32]],
    [[64, 128, 128], [256, 128, 64]],
    [[128, 256, 256], [512, 256, 128]],
    [256, 512, 256]
]


class NetworkRussia(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            layers.PreActivation2d(32, 32, kernel_size=3, padding=1),
        )

        # don't need upsampling here because it will be done in `merge_branches`
        upsample = nn.Sequential  # same as nn.Identity, but supported by older versions
        downsample = partial(nn.MaxPool2d, kernel_size=2)

        self.midline_head = layers.InterpolateToInput(nn.Sequential(
            downsample(),

            # unet
            layers.FPN(
                layers.ResBlock2d, downsample=downsample, upsample=upsample,
                merge=merge_branches, structure=STRUCTURE, kernel_size=3, padding=1
            ),
            # final logits
            layers.PreActivation2d(32, 1, kernel_size=1),
        ), mode='bilinear')

        self.limits_head = layers.InterpolateToInput(nn.Sequential(
            layers.ResBlock2d(32, 32, kernel_size=3, padding=1),
            downsample(),
            layers.ResBlock2d(32, 32, kernel_size=3, padding=1),

            # 2D feature map to 1D feature map
            nn.AdaptiveMaxPool2d((None, 1)),
            layers.Reshape('0', '1', -1),

            nn.Conv1d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        ), mode='linear', axes=0)

    def forward(self, x):
        x = self.init_block(x)
        masks, limits = self.midline_head(x), self.limits_head(x)
        curves = expectation(F.softmax(masks, -1))

        return torch.cat([curves, limits], 1)


# ===== RLDN from UII ===== #


class RLDN(nn.Module):
    def __init__(self, input_size=(400, 304), S=5, init_weights=True):
        super(RLDN, self).__init__()
        # ===== initial param ===== #
        self.h, self.w = input_size; self.s = S

        # ===== backbone vgg16bn with PPM ===== #
        self.bb_1, self.bb_2, self.bb_3, self.bb_4, self.bb_5 = decompose_vgg16()
        self.ppm_1 = PSPModule(32, 32)
        self.ppm_2 = PSPModule(64, 64)
        self.ppm_3 = PSPModule(128, 128)
        self.low_dim_1 = nn.Conv2d(32,  32, 1)
        self.low_dim_2 = nn.Conv2d(64,  32, 1)
        self.low_dim_3 = nn.Conv2d(128, 32, 1)
        self.low_dim_4 = nn.Conv2d(256, 32, 1)
        self.low_dim_5 = nn.Conv2d(256, 32, 1)

        # ===== multi_scale_bidirect_intergate ===== #
        # shallow to deep
        self.fuse_sd_2 = Fuse(32, 32, 'down')
        self.fuse_sd_3 = Fuse(32, 32, 'down')
        self.fuse_sd_4 = Fuse(32, 32, 'down')
        self.fuse_sd_5 = Fuse(32, 32, 'down')
        # deep to shallow
        self.fuse_ds_4 = Fuse(32, 32, 'up')
        self.fuse_ds_3 = Fuse(32, 32, 'up')
        self.fuse_ds_2 = Fuse(32, 32, 'up')
        self.fuse_ds_1 = Fuse(32, 32, 'up')
        # merge
        self.merge_1 = Merge(32,  input_size=input_size)
        self.merge_2 = Merge(32,  input_size=input_size)
        self.merge_3 = Merge(32, input_size=input_size)
        self.merge_4 = Merge(32, input_size=input_size)
        self.merge_5 = Merge(32, input_size=input_size)
        
        # ===== weight_line_intergate ===== #
        self.W_H = nn.Parameter(torch.randn(self.s, self.h, self.w))
        self.intergate = nn.Sequential(nn.Conv2d(self.s, 1, 1), nn.Sigmoid())

        # ===== regression_based_refine ===== #
        self.downsample_a = nn.Sequential(nn.Conv2d(5, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32))
        self.res_blocks_a = BasicBlock(inplanes=5, planes=32, stride=2, downsample=self.downsample_a)
        self.downsample_b = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64))
        self.res_blocks_b = BasicBlock(inplanes=32, planes=64, stride=2, downsample=self.downsample_b)
        self.downsample_c = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128))
        self.res_blocks_c = BasicBlock(inplanes=64, planes=128, stride=2, downsample=self.downsample_c)

        self.softargmax = SoftArgmax(beta=10)
        self.res_blocks = nn.Sequential(*[self.res_blocks_a, self.res_blocks_b, self.res_blocks_c])
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.regress_mid = nn.Sequential(
                        nn.Linear(128 * 4 * 4, 1024),
                        nn.Linear(1024, 600),
                        nn.Linear(600, 400))
        self.regress_anterior_pts  = nn.Linear(200, 2)
        self.regress_posterior_pts = nn.Linear(200, 2)

        # ===== initialize weight ===== #
        if init_weights:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
           

    def forward(self, x):
        bb_features = self._get_bb_features(x)
        y_hat = self._multi_scale_bidirect_intergate(bb_features)
        y_fuse = self._weight_line_intergate(y_hat)
        y_r_hat = self._regression_based_refine(y_hat, y_fuse)

        return y_hat, y_fuse, y_r_hat


    def _get_bb_features(self, x):
        bb_f1 = self.ppm_1(self.bb_1(x))
        bb_f2 = self.ppm_2(self.bb_2(bb_f1))
        bb_f3 = self.ppm_3(self.bb_3(bb_f2))
        bb_f4 = self.bb_4(bb_f3)
        bb_f5 = self.bb_5(bb_f4)

        bb_f1 = self.low_dim_1(bb_f1)
        bb_f2 = self.low_dim_2(bb_f2)
        bb_f3 = self.low_dim_3(bb_f3)
        bb_f4 = self.low_dim_4(bb_f4)
        bb_f5 = self.low_dim_5(bb_f5)

        return [bb_f1, bb_f2, bb_f3, bb_f4, bb_f5]


    def _multi_scale_bidirect_intergate(self, bb_features):
        bb_f1, bb_f2, bb_f3, bb_f4, bb_f5 = bb_features
        #shallow to deep 
        sd_f1 = bb_f1
        sd_f2 = self.fuse_sd_2(sd_f1, bb_f2)
        sd_f3 = self.fuse_sd_3(sd_f2, bb_f3)
        sd_f4 = self.fuse_sd_4(sd_f3, bb_f4)
        sd_f5 = self.fuse_sd_5(sd_f4, bb_f5)
        # deep to shallow
        ds_f5 = bb_f5
        ds_f4 = self.fuse_ds_4(ds_f5, bb_f4)
        ds_f3 = self.fuse_ds_3(ds_f4, bb_f3)
        ds_f2 = self.fuse_ds_2(ds_f3, bb_f2)
        ds_f1 = self.fuse_ds_1(ds_f2, bb_f1)
        # fused feature
        m_f1 = self.merge_1(sd_f1, ds_f1)
        m_f2 = self.merge_2(sd_f2, ds_f2)
        m_f3 = self.merge_3(sd_f3, ds_f3)
        m_f4 = self.merge_4(sd_f4, ds_f4)
        m_f5 = self.merge_5(sd_f5, ds_f5)
        # final concat
        y_hat = torch.cat([m_f1, m_f2, m_f3, m_f4, m_f5], dim=1)

        return y_hat


    def _weight_line_intergate(self, y_hat):
        y_fuse = y_hat * self.W_H
        y_fuse = self.intergate(y_fuse)

        return y_fuse.squeeze(dim=1)


    def _regression_based_refine(self, y_hat, y_fuse):
        y_argmax = self.softargmax(y_fuse)
        y = self.res_blocks(y_hat)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.regress_mid(y)
        a_pts = self.regress_anterior_pts(y[:, :200])
        p_pts = self.regress_posterior_pts(y[:, 200:])
        y_r_hat = torch.cat([y + y_argmax, a_pts, p_pts], dim=1)

        return y_r_hat





