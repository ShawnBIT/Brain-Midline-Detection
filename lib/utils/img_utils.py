# -*- coding: utf-8 -*-
# File: img_utils.py
# Author: Shen Wang <wangshen@pku.edu.cn>

import random

import cv2
import numpy as np 
from PIL import Image
import SimpleITK as sitk

# TBC: random crop and center crop

def center_crop(img, shape):
    h, w = shape[0], shape[1]
    y = (img.shape[0] - h) // 2
    x = (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]


def affine(img, gt, matrix):
    h, w = img.shape[:2]
    img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR)
    if isinstance(gt, list) or isinstance(gt, tuple):  
        gt_1, gt_2 = gt
        gt_1 = cv2.warpAffine(gt_1, matrix, (w, h), flags=cv2.INTER_NEAREST)
        gt_2 = cv2.warpAffine(gt_2, matrix, (w, h), flags=cv2.INTER_NEAREST)
        gt = (gt_1, gt_2)
    else:
        gt = cv2.warpAffine(gt, matrix, (w, h), flags=cv2.INTER_NEAREST)

    return img, gt


def resize(img, gt, size):
    # because of most of the data, its shape is 512*512
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if isinstance(gt, list) or isinstance(gt, tuple):  
        gt_1, gt_2 = gt
        gt_1 = cv2.resize(gt_1, (size, size), interpolation=cv2.INTER_NEAREST)
        gt_2 = cv2.resize(gt_2, (size, size), interpolation=cv2.INTER_NEAREST)
        gt = (gt_1, gt_2)
    else:
        gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)

    return img, gt
    

def translate(img, gt, x, y): 
    h, w = img.shape[:2]
    translate_matrix = np.float32([[1, 0, x], [0, 1, y]]) 
    img = cv2.warpAffine(img, translate_matrix, (w, h), flags=cv2.INTER_LINEAR)
    if isinstance(gt, list) or isinstance(gt, tuple):  
        gt_1, gt_2 = gt
        gt_1 = cv2.warpAffine(gt_1, translate_matrix, (w, h), flags=cv2.INTER_NEAREST)
        gt_2 = cv2.warpAffine(gt_2, translate_matrix, (w, h), flags=cv2.INTER_NEAREST)
        gt = (gt_1, gt_2)
    else:
        gt = cv2.warpAffine(gt, translate_matrix, (w, h), flags=cv2.INTER_NEAREST)
    
    return img, gt


def rotate(img, gt, angle=20):
    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    if isinstance(gt, list) or isinstance(gt, tuple):  
        gt_1, gt_2 = gt
        gt_1 = cv2.warpAffine(gt_1, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        gt_2 = cv2.warpAffine(gt_2, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        gt = (gt_1, gt_2)
    else:
        gt = cv2.warpAffine(gt, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    return img, gt


def random_scale(img, gt, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if isinstance(gt, list) or isinstance(gt, tuple):  
        gt_1, gt_2 = gt
        gt_1 = cv2.resize(gt_1, (sw, sh), interpolation=cv2.INTER_NEAREST)
        gt_2 = cv2.resize(gt_2, (sw, sh), interpolation=cv2.INTER_NEAREST)
        gt = (gt_1, gt_2)
    else:
        gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt


def random_rotate(img, gt, rotation=20):
    angle = random.random() * 2 * rotation - rotation
    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    if isinstance(gt, list) or isinstance(gt, tuple):  
        gt_1, gt_2 = gt
        gt_1 = cv2.warpAffine(gt_1, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        gt_2 = cv2.warpAffine(gt_2, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        gt = (gt_1, gt_2)
    else:
        gt = cv2.warpAffine(gt, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    return img, gt


def random_mirror(img, gt):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        if isinstance(gt, list) or isinstance(gt, tuple):
            gt_1, gt_2 = gt
            gt_1 = cv2.flip(gt_1, 1)
            gt_2 = cv2.flip(gt_2, 1)
            gt = (gt_1, gt_2)
        else:
            gt = cv2.flip(gt, 1)

    return img, gt


def random_gaussian_blur(img, gaussian_kernels):
    gauss_size = random.choice(gaussian_kernels)
    if gauss_size > 1:
        # do the gaussian blur
        img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 0)

    return img


def random_brightness(img, brightness=20):
    assert brightness >= 0
    brightness_value = random.random() * 2 * brightness - brightness
    contrast = 1    
    bri_image = cv2.addWeighted(img, contrast, img, 0, brightness_value)
    
    return bri_image


def normalize(img, mean=None, std=None):
    img = img.astype(np.float32) / 255.0
    if mean is not None and std is not None:
        img = img - mean
        img = img / std

    return img


def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                             border_mode, value=value)

    return img


########## old
def rotateMatrix(R_x, R_y, R_z, zoom):
    trans_M =np.dot(rotateZMatrix(R_z, zoom[2]), rotateYMatrix(R_y, zoom[1]))
    trans_M =np.dot(trans_M, rotateXMatrix(R_x, zoom[0]))
    return trans_M

    
def rotation3d_point(shape, point, R_x = 0, R_y = 0, R_z = 0 , zoom = (1, 1, 1), shear = (0, 0, 0, 0, 0, 0), interpolator=sitk.sitkLinear):
    '''
    img: (z, y, x)
    point: tuple (z, y, x)
    '''
    point = point[::-1]
    centre_in=0.5*np.array(shape[::-1])
    rot = np.dot(rotateMatrix(R_x, R_y, R_z, zoom), shearMatrix(shear))
    
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(np.reshape(rot.T, -1)[::-1])
    centering_transform = sitk.TranslationTransform(3)
    centering_transform.SetOffset(np.array(affine.GetInverse().TransformPoint(centre_in) - centre_in))
    
    all_transform = sitk.Transform(affine)
    all_transform.AddTransform(centering_transform)    
    trans_point = all_transform.TransformPoint(point)[::-1]
    trans_point = (int(trans_point[1]), int(trans_point[2]))
    return trans_point

def successive_process(midline):
    #from IPython import embed; embed()
    sucess_midline = np.zeros_like(midline.astype('uint8'))
    contours, hierarchy = cv2.findContours(midline.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rotate_midline, contours, -1, 1, 1)

    return sucess_midline


def rotate_successive(midline, angle):
    #from IPython import embed; embed()
    rotate_midline = np.zeros_like(midline.astype('uint8'))
    contours, hierarchy = cv2.findContours(midline.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out_cnt = []
    for cnt in contours:
        cnt_rot = np.zeros_like(cnt)
        for pix, point in enumerate(cnt):
            ''' point: x,y to y, x '''
    #         point_zyx = np.zeros_like() 
            y , x = point[0]        
            rot_point = rotation3d_point((1, )+midline.shape, (0, int(y), int(x)), R_x = 0, R_y = 0, R_z = angle, zoom = (1, 1, 1), shear = (0, 0, 0, 0, 0, 0), interpolator=sitk.sitkLinear)
            cnt_rot[pix] = rot_point
        out_cnt.append(cnt_rot)    
    cv2.drawContours(rotate_midline, out_cnt, -1, 1, 1)
    return rotate_midline


def rotate_successive_3D(midline_3D, angle):
    new_midline = np.zeros(midline_3D.shape)
    for ii in range(midline_3D.shape[0]):
        midline_slice = midline_3D[ii]
        new_midline[ii] = rotate_successive(midline_slice,angle)

    return new_midline


# transform: 平移操作
def shift_3D(pred, center, img_center=(256, 256)):
    pred_new = np.zeros(pred.shape)
    for ii in range(pred.shape[0]):
        pred_slice = pred[ii]
        pred_slice = shift_2D(pred_slice, center, img_center)
        pred_new[ii] = pred_slice
    return pred_new


def shift_2D(pred, center, img_center):
    cx, cy = center
    img_x, img_y = img_center
    img = Image.fromarray(pred)
    img = img.transform(size=img.size,method=Image.AFFINE, data=(1,0,-(cx-img_x),0,1,-(cy-img_y)))
    return np.array(img)







