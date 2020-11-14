import cv2
import numpy as np
import math
from collections import OrderedDict
import torch
import copy
import torch.nn.functional as F

from .russia import limits_to_mask, curves2midline 

# 中线的激发方式：Max
class Mid_Threshold(object):
    
    def __init__(self, shape):
        self.shape = shape

    def draw_point(self, target, y_min):
        final = np.zeros(self.shape)
        for point in target:
            y, x = point
            final[y+y_min][x] = 1
           
        return final

    def get_midline(self, prob_slice):
        prob_width5 = prob_slice
        midline = (prob_slice > 0.5).astype('uint8')
        y_range, x_range = np.nonzero(midline)
        prob = prob_width5[y_range[0]:y_range[-1],:]
        target_points = self.find_path(prob)
        final = self.draw_point(target_points, y_range[0])

        return final

    def get_midline_3D(self, prob_3D):
        new_midline = np.zeros(prob_3D.shape)
        for z in range(prob_3D.shape[0]):
            prob_slice = prob_3D[z]
            midline_slice = (prob_slice > 0.5).astype('uint8')
            if midline_slice.sum() > 0:
                new_midline[z] = self.get_midline(prob_slice)
        return new_midline
    
    def find_path(self, prob):
        target_points = []
        y_range, x_range = np.where(prob > 0.5)
        for (y, x) in zip(y_range, x_range):
            target_points.append((y, x))

        return target_points



# 中线的激发方式：DP
class Mid_DP(object):
    
    def __init__(self, shape):
        self.shape = shape
        self.h, self.w = shape[:2]

    def MaxPathSum(self, grid, x_ymin):
        x, y = len(grid), len(grid[0])
        dp = [[0 for _ in range(y)] for _ in range(x)]
        for i in range(self.w):
            dp[0][i] = grid[0][i] if i == x_ymin else 0

        for i in range(1, x):
            for j in range(y):
                if j == 0:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j+1]) + grid[i][j]
                elif j == y - 1:
                    dp[i][j] = max(dp[i-1][j-1], dp[i-1][j]) + grid[i][j]
                else:
                    dp[i][j] = max(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1]) + grid[i][j]

        return dp

    def find_path(self, dp, x_ymax):
        i,j = len(dp)-1, x_ymax
        target_list = []
        while i != 0 and j < self.w - 1 and j > 1:
            target_list.append((i,j))
            if max(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1]) == dp[i-1][j-1]:
                j = j-1
            elif max(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1]) == dp[i-1][j]:
                pass
            elif max(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1]) == dp[i-1][j+1]:
                 j = j+1
            i -= 1
        return target_list

    def draw_point(self, target, y_min):
        final = np.zeros(self.shape)
        for point in target:
            y, x = point
            final[y+y_min][x] = 1
            
        return final

    def get_midline(self, prob_slice):
        prob_width5 = prob_slice
        midline = (prob_slice > 0.5).astype('uint8')
        y_range, x_range = np.nonzero(midline)
        prob_dp = prob_width5[y_range[0]:y_range[-1],:]
        dp = self.MaxPathSum(prob_dp, x_range[0])
        target_points = self.find_path(dp, x_range[-1])
        final = self.draw_point(target_points, y_range[0])

        return final
    
    def get_midline_3D(self, prob_3D):
        new_midline = np.zeros(prob_3D.shape)
        for z in range(prob_3D.shape[0]):
            prob_slice = prob_3D[z]
            midline_slice = (prob_slice > 0.5).astype('uint8')
            if midline_slice.sum() > 2:
                new_midline[z] = self.get_midline(prob_slice)
        return new_midline
    

# 中线的激发方式：Max
class Mid_Argmax(object):
    
    def __init__(self, shape):
        self.shape = shape

    def draw_point(self, target, y_min):
        final = np.zeros(self.shape)
        for point in target:
            y, x = point
            final[y+y_min][x] = 1
           
        return final

    def get_midline(self, prob_slice):
        prob_width5 = prob_slice
        midline = (prob_slice > 0.5).astype('uint8')
        y_range, x_range = np.nonzero(midline)
        prob = prob_width5[y_range[0]:y_range[-1],:]
        target_points = self.find_path(prob)
        final = self.draw_point(target_points, y_range[0])

        return final

    def get_midline_3D(self, prob_3D):
        new_midline = np.zeros(prob_3D.shape)
        for z in range(prob_3D.shape[0]):
            prob_slice = prob_3D[z]
            midline_slice = (prob_slice > 0.5).astype('uint8')
            if midline_slice.sum() > 0:
                new_midline[z] = self.get_midline(prob_slice)
        return new_midline
    
    def find_path(self, prob):
        target_points = []
        for y in range(prob.shape[0]):
            x = np.argmax(prob[y])
            target_points.append((y,x))
        return target_points


# 中线宽度从1扩充为3
def midline_expand(prob_map):
    pred = np.zeros(prob_map.shape)
    for z in range(prob_map.shape[0]):
        prob_slice = prob_map[z]
        for y in range(prob_slice.shape[0]):
            prob_horizon = prob_slice[y]
            max_x = np.argmax(prob_horizon)
            if max_x != 0 and max_x != 303:
                pred[z][y][max_x - 1] = 1
                pred[z][y][max_x] = 1
                pred[z][y][max_x + 1] = 1

    return pred


def get_ideal_midline(mid):
    ideal_mid = np.zeros(mid.shape, dtype=np.uint8)
    y_range, x_range = np.where(mid > 0)
    top_point, bottom_point = (x_range[0], y_range[0]), (x_range[-1], y_range[-1])
    cv2.line(ideal_mid, top_point, bottom_point, 255, 1)
    
    return ideal_mid


class AlignInfo(object):
    def __init__(self):
        pass
    
    def get_align_info(self, mid):
        y_range, x_range = np.nonzero(mid) 
        assert len(y_range) > 0 and len(x_range) > 0

        up_x, up_y = x_range[0], y_range[0] 
        down_x, down_y = x_range[-1], y_range[-1]
        cx = (up_x + down_x)//2
        cy = (up_y + down_y)//2

        if down_x == up_x:
            angle = 0
        else:  
            k = (down_y - up_y)*1.0/(down_x - up_x) #斜率
            angle = np.arctan(k)
            if angle < 0:
                angle = 90 + angle*180/math.pi
            else:
                angle = -90 + angle*180/math.pi

        return angle, cx, cy
    

    def get_affine_matrix(self, mid):
        h, w = mid.shape
        angle, cx, cy = self.get_align_info(mid)
        dx, dy = w//2-cx, h//2-cy
        trans_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        rotate_matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        trans_matrix_R3 = np.zeros((3, 3))
        trans_matrix_R3[:2, :] = trans_matrix
        trans_matrix_R3[2][2] = 1
        rotate_matrix_R3 = np.zeros((3, 3))
        rotate_matrix_R3[:2, :] = rotate_matrix
        rotate_matrix_R3[2][2] = 1
        affine_matrix_R3 = np.matmul(rotate_matrix_R3, trans_matrix_R3)
        affine_matrix = affine_matrix_R3[:2, :]
        meta_info = (angle, dx, dy)

        return affine_matrix_R3, affine_matrix, meta_info


def get_reg_gt(mid):
    cord_dict = OrderedDict()
    y_range, x_range = np.nonzero(mid)
    h, w = mid.shape
    
    for y, x in zip(y_range, x_range):
        if y not in cord_dict:
            cord_dict[y] = [x, 1]
        else:
            cord_dict[y][0] += x
            cord_dict[y][1] += 1 

    reg_gt = [0 for _ in range(h)]
    for y in range(h):
        if y in cord_dict:
            reg_gt[y] = cord_dict[y][0]/cord_dict[y][1]
        else:
            reg_gt[y] = (w-1)*1.0/2

    reg_gt.extend([x_range[0], y_range[0], x_range[-1], y_range[-1]])
    
    return np.array(reg_gt)



def regvalue2mid(regvalue):
    reg_mid = regvalue[:, :-2]
    mid_range = regvalue[:, -2:]
    batch, height = reg_mid.shape
    width = 304
    res_mid = torch.zeros(batch, height, width)
    for b in range(batch):
        reg_v = reg_mid[b]
        y_min, y_max = mid_range[b]
        for h in range(height):
            if h >= y_min and h <= y_max:
                x_coord = torch.round(reg_v[h].detach()).long()
                res_mid[b][h][x_coord] = 1

    return res_mid


def mid2coord_RLDN(mid, heatmap_param):
    delta, base_val = heatmap_param
    y_range, x_range = np.nonzero(mid)
    assert len(y_range) >= 2
    h, w = mid.shape
    x_coords = []
    for y in range(h):
        x = np.argmax(mid[y]) if y in y_range else (w-1)*1.0/2 # attention softargmax, float
        x_coords.append(x)

    flag_pts = [y_range[0], x_range[0], y_range[-1], x_range[-1]]
    x_coords.extend(flag_pts)
    x_coords = np.array(x_coords, dtype=np.float32)

    x_coords2 = []
    for y in range(h):
        x = np.argmax(mid[y]) if y in y_range else 0 # attention softargmax, float
        x_coords2.append(x)
    x_coords2 = np.array(x_coords2, dtype=np.float32)

    neg_mask = np.zeros(mid.shape)
    neg_mask[:y_range[0], :] = 1
    neg_mask[y_range[-1]+1:, :] = 1
    neg_mask = np.array(neg_mask, dtype=np.bool)

    h, w = mid.shape
    coord_map = np.repeat(np.arange(w), axis=0, repeats=h).reshape(w, h).T
    minus_val = np.repeat(x_coords2, axis=0, repeats=w).reshape(h, w)
    dist_map = np.abs(coord_map - minus_val) - 1

    heatmap = base_val * np.exp(-(dist_map ** 2)/ (2 * delta ** 2))
    heatmap[neg_mask] = 0

    pos_mask = np.array(dist_map < 1, dtype = np.bool)
    dist_map = dist_map / w
    dist_map[neg_mask] = 1
    dist_map[pos_mask] = 1
    dist_map = np.array(dist_map, dtype=np.float32)

    mask = (mid != 0).astype(np.float32)
    num_positive = np.sum(mask).astype(np.float32)
    num_negative = h * w - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
   
    return x_coords, dist_map, mask, heatmap


def mid2coord_Russia(mid, heatmap_param):
    delta, base_val = heatmap_param
    y_range, x_range = np.nonzero(mid)
    assert len(y_range) >= 2
    h, w = mid.shape
    x_coords = []
    for y in range(h):
        x = np.argmax(mid[y]) if y in y_range else np.nan # attention softargmax, float
        x_coords.append(x)
    x_coords = np.array(x_coords, dtype=np.float32)

    x_coords2 = []
    for y in range(h):
        x = np.argmax(mid[y]) if y in y_range else 0 # attention softargmax, float
        x_coords2.append(x)
    x_coords2 = np.array(x_coords2, dtype=np.float32)

    neg_mask = np.zeros(mid.shape)
    neg_mask[:y_range[0], :] = 1
    neg_mask[y_range[-1]+1:, :] = 1
    neg_mask = np.array(neg_mask, dtype=np.bool)

    h, w = mid.shape
    coord_map = np.repeat(np.arange(w), axis=0, repeats=h).reshape(w, h).T
    minus_val = np.repeat(x_coords2, axis=0, repeats=w).reshape(h, w)
    dist_map = np.abs(coord_map - minus_val)

    heatmap = base_val * np.exp(-(dist_map ** 2)/ (2 * delta ** 2))
    #heatmap[heatmap <= 1] = 1
    heatmap[neg_mask] = 0

    dist_map[dist_map == 0] = 1
    dist_map = 10 / dist_map 
    dist_map[neg_mask] = 0
    dist_map = np.array(dist_map, dtype=np.float32)


    mask = (mid != 0).astype(np.float32)
    num_positive = np.sum(mask).astype(np.float32)
    num_negative = h * w - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)

    return x_coords, dist_map, mask, heatmap


def mid2coord(mid, heatmap_param, way):
    if way == 'rldn':
        return mid2coord_RLDN(mid, heatmap_param)
    elif way == 'russia' or way == 'ours':
        return mid2coord_Russia(mid, heatmap_param)


 
def get_curves_gt_limit(gt_limits_mask, pred_limits_mask, curves):
    curves_gt_limit = pred_limits_mask * curves
    # make sure the same length
    for b in range(curves_gt_limit.shape[0]):
        h = curves_gt_limit[b].shape[0]
        y_range = np.nonzero(curves_gt_limit[b])[0]
        y_min, y_max = y_range[0], y_range[-1]
        for y in range(h):
            if y < y_min:
                curves_gt_limit[b][y] = curves_gt_limit[b][y_min]
            elif y > y_max:
                curves_gt_limit[b][y] = curves_gt_limit[b][y_max]


    return gt_limits_mask * curves_gt_limit




def limits2mask(curves, limits):
    mask = np.zeros(curves.shape, dtype=np.float32)
    for b in range(curves.shape[0]):
        y_min = int(np.round(limits[b][0]))
        y_max = int(np.round(limits[b][2]))
        mask[b][y_min:y_max+1] = 1 

    return mask


def pred2midline(pred, gt_curves, model_name):
    if model_name == 'russia':
        gt_limits_mask = ~torch.isnan(gt_curves)
        curves, limits = pred[:, 0], pred[:, 1]
        curves = curves.cpu().data.numpy()
        limits = limits.cpu().data.numpy()
        gt_limits_mask = gt_limits_mask.cpu().data.numpy()
        mask = limits_to_mask(limits)
        curves_real_limit = mask * curves
        curves_gt_limit = gt_limits_mask * curves
        pred_midline_real_limit = curves2midline(curves_real_limit)
        pred_midline_gt_limit = curves2midline(curves_gt_limit)
        return pred_midline_real_limit, pred_midline_gt_limit

    elif model_name == 'rldn':
        y_hat, y_fuse, y_r = pred
        y_r = y_r.cpu().data.numpy()
        x_coords = y_r[:, :400]
        limits = y_r[:, 400:]
        gt_x_coords = gt_curves[:, :400]
        gt_limits = gt_curves[:, 400:]
        real_mask = limits2mask(x_coords, limits)
        gt_mask = limits2mask(gt_x_coords, gt_limits)
        curves_real_limit = real_mask * x_coords 
        curves_gt_limit = gt_mask * x_coords
        pred_midline_real_limit = curves2midline(curves_real_limit)
        pred_midline_gt_limit = curves2midline(curves_gt_limit)

        return pred_midline_real_limit, pred_midline_gt_limit

    elif 'our' in model_name:
        prob_map, curves_limits = pred
        curves, limits = curves_limits[:, 0], curves_limits[:, 1]
        gt_limits_mask = ~torch.isnan(gt_curves)
        curves = curves.cpu().data.numpy()
        limits = limits.cpu().data.numpy()
        gt_limits_mask = gt_limits_mask.cpu().data.numpy()
        mask = limits_to_mask(limits)
        curves_real_limit = mask * curves
        curves_gt_limit = gt_limits_mask * curves
        pred_midline_real_limit = curves2midline(curves_real_limit)
        pred_midline_gt_limit = curves2midline(curves_gt_limit)

        return pred_midline_real_limit, pred_midline_gt_limit
    
    elif model_name == 'mdnet':
        prob_map_1, prob_map_2, prob_map_3, prob_map_4 = pred
        mid_gen = Mid_DP((400, 304))
        prob_map_1 = prob_map_1.cpu().data.numpy()
        midline = mid_gen.get_midline_3D(prob_map_1)
        pred_midline_real_limit = midline
        curves = midline2curves_mdnet(midline)
        gt_limits_mask = ~torch.isnan(gt_curves)
        curves_gt_limit = gt_limits_mask.numpy() * curves
        pred_midline_gt_limit = curves2midline(curves_gt_limit)

        return pred_midline_real_limit, pred_midline_gt_limit


def midline2curves_mdnet(midline):
    N, h, w = midline.shape
    curves = np.zeros((N, h), dtype=np.float32)
    for i in range(N):
        mid = midline[i]
        y_range, x_range = np.nonzero(mid)
        if len(y_range) <=1:
            continue
        y_min, y_max = y_range[0], y_range[-1]
        curves[i][:y_min] = x_range[0]
        curves[i][y_min:y_max+1] = x_range
        curves[i][y_max+1:] = x_range[-1]
    
    return curves


def midline2curves(mid):
    h, w = mid.shape
    curves = np.zeros(h, dtype=np.float32)
    curves[:] = np.nan
    y_range, x_range = np.nonzero(mid)
    if len(y_range) < 2:
        return curves
    y_min, y_max = y_range[0], y_range[-1]
    curves[y_min:y_max+1] = x_range
    
    return curves





