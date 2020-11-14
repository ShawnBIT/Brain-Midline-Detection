import numpy as np
import os
import os.path as osp
import cv2
import time
import math
import scipy.misc as misc
from skimage import measure


# Image Operation
def image_tensor_resize(img_tensor, size=(-1, -1, -1), method='B'):
    new_d, new_h, new_w = size
    d, h, w = img_tensor.shape
    new_shape = (new_d, new_h, new_w)
    if d == new_d and h == new_h and w == new_w:
        return img_tensor
    tmp_img_tensor = np.zeros((img_tensor.shape[0], new_shape[1], new_shape[2]), dtype=img_tensor.dtype)
    new_img_tensor = np.zeros(new_shape, dtype=img_tensor.dtype)
    for idx in range(img_tensor.shape[0]):
        if method == 'B':
            tmp_img_tensor[idx, :, :] = cv2.resize(img_tensor[idx, :, :], (new_shape[2], new_shape[1]), interpolation=cv2.INTER_CUBIC)
        elif method == 'L':
            tmp_img_tensor[idx, :, :] = cv2.resize(img_tensor[idx, :, :], (new_shape[2], new_shape[1]), interpolation=cv2.INTER_LINEAR)
        else:
            tmp_img_tensor[idx, :, :] = cv2.resize(img_tensor[idx, :, :], (new_shape[2], new_shape[1]), interpolation=cv2.INTER_NEAREST)

    for idx in range(new_shape[1]):
        if method == 'B':
            new_img_tensor[:, idx, :] = cv2.resize(tmp_img_tensor[:, idx, :], (new_shape[2], new_shape[0]),
                                                   interpolation=cv2.INTER_CUBIC)
        elif method == 'L':
            new_img_tensor[:, idx, :] = cv2.resize(tmp_img_tensor[:, idx, :], (new_shape[2], new_shape[0]),
                                                   interpolation=cv2.INTER_LINEAR)
        else:
            new_img_tensor[:, idx, :] = cv2.resize(tmp_img_tensor[:, idx, :], (new_shape[2], new_shape[0]),
                                                   interpolation=cv2.INTER_NEAREST)
    return new_img_tensor


def imsave_tensor(img_tensor, src_dir, direction=0, ext='png'):
    for idx in range(img_tensor.shape[direction]):
        if direction == 0:
            cv2.imwrite(src_dir + '/%03d.' % idx + ext, img_tensor[idx, :, :])
        elif direction == 1:
            cv2.imwrite(src_dir + '/%03d.' % idx + ext, img_tensor[:, idx, :])
        elif direction == 2:
            cv2.imwrite(src_dir + '/%03d.' % idx + ext, img_tensor[:, :, idx])


def gray2rgbimage(image):
    a,b = image.shape
    new_img = np.ones((a,b,3))
    new_img[:,:,0] = image.reshape((a,b)).astype('uint8')
    new_img[:,:,1] = image.reshape((a,b)).astype('uint8')
    new_img[:,:,2] = image.reshape((a,b)).astype('uint8')
    return new_img


class visualize_grid(object):
    def __init__(self, ny, nx, grid_size):
        self.ny = ny
        self.nx = nx
        self.grid_size = grid_size
        self.img = np.zeros((ny*grid_size, nx*grid_size, 3), dtype = np.uint8)

    def draw(self, index, sub_img):
        if(index == 0):
            self.img[...] = 0
        y = int(index / self.nx)
        x = int(index % self.nx)
        start_y = y*self.grid_size
        end_y = (y+1)*self.grid_size
        start_x = x*self.grid_size
        end_x = (x+1)*self.grid_size
        self.img[start_y:end_y, start_x:end_x] = sub_img

    def get_image(self):
        return self.img

def visualize_result(imt, ant, pred, save_dir,view='Z',gap_num=8):

    if view == 'Z':
        vis_size = (imt.shape[0],256,256)
        imt = image_tensor_resize(imt,vis_size,method='B')
        if ant is not None:
            ant = image_tensor_resize(ant,vis_size,method='N').astype('uint8')
        pred = image_tensor_resize(pred,vis_size,method='N').astype('uint8')
    elif view == 'Y':
        vis_size = (imt.shape[1],256,256)
        imt = image_tensor_resize(imt.transpose(2,0,1),vis_size,method='B')
        ant = image_tensor_resize(ant.transpose(2,0,1),vis_size,method='N').astype('uint8')
        pred = image_tensor_resize(pred.transpose(2,0,1),vis_size,method='N').astype('uint8')
    elif view == 'X':
        vis_size = (imt.shape[1],256,256)
        imt = image_tensor_resize(imt.transpose(1,0,2),vis_size,method='B')
        ant = image_tensor_resize(ant.transpose(1,0,2),vis_size,method='N').astype('uint8')
        pred = image_tensor_resize(pred.transpose(1,0,2),vis_size,method='N').astype('uint8')

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    for idx in range(pred.shape[0]):
        gap = 1 if view == 'Z' else gap_num
        if idx%gap == 0:
            binary = pred[idx]
            image, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            temp_img1 = gray2rgbimage(imt[idx])
            color=(0,0,255)
            cv2.drawContours(temp_img1, contours, -1, color, 1)        
            if(ant is None):
                cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((temp_img1, gray2rgbimage(imt[idx]))).astype('uint8'))
            else:
                # Draw GT
                binary = ant[idx]
                image, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                temp_img2 = gray2rgbimage(imt[idx])
                cv2.drawContours(temp_img2, contours, -1, color, 1)
                cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((temp_img1, temp_img2)).astype('uint8'))

def vis_boundary(imt, ant, pred, save_dir,is_background=True):
    # vis_size = (imt.shape[0], 512, 512)
    # imt = image_tensor_resize(imt,vis_size,method='B')
    # if ant is not None:
    #     ant = image_tensor_resize(ant,vis_size,method='N').astype('uint8')
    # pred = image_tensor_resize(pred,vis_size,method='N').astype('uint8')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    vis_imt = []
    for idx in range(pred.shape[0]):
        img = gray2rgbimage(imt[idx])
        img = cv2.putText(img, '', (150,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # white title
        pred_img = draw_img(imt[idx], pred[idx],title='' , is_boundary = True,is_background=is_background)
        vis_imt.append(pred_img)
        if(ant is None):
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((img, pred_img)).astype('uint8'))
        else:
            ant_img = draw_img(imt[idx], ant[idx],title='' , is_boundary = True,is_background=is_background)
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((img, pred_img, ant_img)).astype('uint8'))


def vis_multi_class(imt, ant, pred, save_dir,n_class=3,is_boundary=False):
    vis_size = (imt.shape[0],256,256)
    imt = image_tensor_resize(imt,vis_size,method='B')
    if ant is not None:
        ant = image_tensor_resize(ant,vis_size,method='N').astype('uint8')
    pred = image_tensor_resize(pred,vis_size,method='N').astype('uint8')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    for idx in range(pred.shape[0]):
        img = gray2rgbimage(imt[idx])
        pred_img = draw_img(imt[idx], pred[idx],title='Model Pred',is_boundary=is_boundary,n_class=n_class)
        if(ant is None):
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((img, pred_img)).astype('uint8'))
        else:
            # Draw GT
            ant_img = draw_img(imt[idx], ant[idx],title='Ground Truth',is_boundary=is_boundary, n_class=n_class)            
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((img, pred_img, ant_img)).astype('uint8'))

def vis_multi_class_single(imt, pred, title = None, save_dir = None):
    vis_size = (imt.shape[0],256,256)
    imt = image_tensor_resize(imt, vis_size, method='B')
    pred = image_tensor_resize(pred, vis_size, method='N').astype('uint8')

    if save_dir is not None and not osp.exists(save_dir):
        os.makedirs(save_dir)
    vis_imt = []
    for idx in range(pred.shape[0]):
        pred_img = draw_img(imt[idx], pred[idx], title)
        vis_imt.append(pred_img)
        if(save_dir is not None):
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), temp_img.astype('uint8'))
    return vis_imt

def vis_boundary_single(imt, pred, title = None, save_dir = None):
    vis_size = (imt.shape[0],256,256)
    imt = image_tensor_resize(imt, vis_size, method='B')
    pred = image_tensor_resize(pred, vis_size, method='N').astype('uint8')

    if save_dir is not None and not osp.exists(save_dir):
        os.makedirs(save_dir)
    vis_imt = []
    for idx in range(pred.shape[0]):
        pred_img = draw_img(imt[idx], pred[idx], title, is_boundary = True)
        vis_imt.append(pred_img)
        if(save_dir is not None):
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), temp_img.astype('uint8'))
    return vis_imt

def draw_img(img, seg, title = None, is_boundary = False, n_class=4, is_background=True):
    label_set = [i+1 for i in range(n_class)]
    color_set = {
                    1:(0,0,255),
                    2:(0,255,0),
                    3:(255,0,0),
                    4:(0,255,255),
                }

    img = gray2rgbimage(img) 
    if not is_background:
        img[img > -1] = 0 
    if(title is not None):
        img = cv2.putText(img, title, (150,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # white title
    for draw_label in label_set:
        if(is_boundary):
            img[:, :, 0][seg == draw_label] = color_set[draw_label][0]
            img[:, :, 1][seg == draw_label] = color_set[draw_label][1]
            img[:, :, 2][seg == draw_label] = color_set[draw_label][2]
        else:
            seg_label = (seg == draw_label).astype('uint8')
            image, contours, hierarchy = cv2.findContours(seg_label,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            color= color_set[draw_label]
            cv2.drawContours(img, contours, -1, color, 1)     #粗细    
    return img

def visualize_single_pred(imt, pred, save_dir = None):
    if save_dir is not None and not osp.exists(save_dir):
        os.makedirs(save_dir)
    vis_imt = []
    for idx in range(pred.shape[0]):
        binary = pred[idx]
        image, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        temp_img = gray2rgbimage(imt[idx])
        color=(0,0,255)
        cv2.drawContours(temp_img, contours, -1, color, 1)        
        temp_img = cv2.resize(temp_img, (512, 512), interpolation=cv2.INTER_LINEAR)        
        vis_imt.append(temp_img)
        if(save_dir is not None):
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), temp_img.astype('uint8'))
    return vis_imt


# mask 2 midline

def find_contour(mask,label):
    img = np.zeros(mask.shape)
    seg_label = (mask == label).astype('uint8')
    image, contours, hierarchy = cv2.findContours(seg_label,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1,255, 3) 
    img[img != 0] = 1
    return img

def mask2midline(mask):
    mid = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        mask_i = mask[i]
        img_1 = find_contour(mask_i,1)
        img_2 = find_contour(mask_i,2)
        mid_i =  (img_1+img_2 == 2).astype('uint8')
        mid[i] = mid_i

    return mid








