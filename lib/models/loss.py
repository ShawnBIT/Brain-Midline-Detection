import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Criterion(object):
    def __init__(self, loss_name):
        self.loss_name = loss_name
        self.loss_family = {
                        'ce': ce_loss,
                        'ce_dice': ce_dice_loss,
                        'mse': mse_loss,
                        'L1': L1_loss
                        }

    def get_loss_func(self):
        if self.loss_name in self.loss_family.keys():
            return self.loss_family[self.loss_name]
        else:
            raise NameError("Unknow Loss type!")

    def compute_loss(self, output, target):
        loss_func = self.get_loss_func()
        Loss = loss_func(output, target)

        return Loss


def L1_loss(output, target):
    criterion_1 = nn.L1Loss()
    criterion_1.cuda()
    loss = criterion_1(output, target)

    return loss


def mse_loss(output, target):
    criterion_1 = nn.MSELoss()
    criterion_1.cuda()
    loss = criterion_1(output, target)

    return loss


def ce_loss(output, target):
    criterion_1 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 10]).float().cuda())
    criterion_1.cuda()
    loss = criterion_1(output, target)

    return loss


def dice_loss(output, target):
    prob_seg = F.softmax(output, dim=1)
    criterion_1 = DiceLoss(class_num=2)
    criterion_1.cuda()
    loss = criterion_1(prob_seg, target)  

    return loss


def ce_dice_loss(output, target):
    loss_1 = ce_loss(output, target)
    loss_2 = dice_loss(output, target)

    return loss_1 + loss_2



def wbce_loss(prob, target, weights):
    loss_fn = nn.BCELoss(weight=weights)
    loss = loss_fn(prob, target.to(dtype=prob.dtype))

    return loss



def dice_sigmoid_loss(prob, target):
    criterion_1 = DiceLossSigmoid(class_num=2)
    criterion_1.cuda()
    loss = criterion_1(prob, target)  

    return loss


def cross_entropy_loss(prediction, label):
    #print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = F.binary_cross_entropy(prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.mean(cost)



class DiceLoss(nn.Module):  
    def __init__(self, class_num=2, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = input[:,i,:,:]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += 1 - dice
        dice_loss = Dice / (self.class_num - 1)
        
        return dice_loss


class DiceLossSigmoid(nn.Module):  
    def __init__(self, class_num=2, smooth=1):
        super(DiceLossSigmoid, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        intersect = (input * target.float()).sum()
        union = torch.sum(input) + torch.sum(target.float())
        dice = (2 * intersect + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        return dice_loss



def get_total_loss(cfg, pred, targets, epoch):
    model_name = cfg.model_param.model_name
    ce_weight = cfg.loss_param.ce_weights
    seg_type = cfg.loss_param.seg_type
    with_dice = cfg.loss_param.with_dice
    with_connect = cfg.loss_param.with_connect

    targets_coord, targets_mid_3, pos_weight, nce_weight, heatmap = targets
    class_weight = Variable(torch.FloatTensor(ce_weight)).cuda() # 这里正例比较少，因此权重要大一些
    wce_weight = class_weight[targets_mid_3.long()] # (3, 4)

    if model_name == 'russia':
        # the model has 2 `heads`
        assert pred.shape[1] == 2, pred.shape
        curves, limits = pred[:, 0], pred[:, 1]
        limits_mask = ~torch.isnan(targets_coord)
        loss_1 = F.binary_cross_entropy_with_logits(limits, limits_mask.to(dtype=limits.dtype))
        loss_2 = F.mse_loss(curves[limits_mask], targets_coord[limits_mask])  # or l1_loss
        loss = loss_1 + loss_2
        loss_3 = loss_2
    elif model_name == 'rldn':
        y_hat, y_fuse, y_r_hat = pred
        weights = wce_weight if seg_type == 'wce' else nce_weight
        loss_1  = wbce_loss(y_hat[:, 0, :, :], targets_mid_3, weights=weights)
        loss_1 += wbce_loss(y_hat[:, 1, :, :], targets_mid_3, weights=weights)
        loss_1 += wbce_loss(y_hat[:, 2, :, :], targets_mid_3, weights=weights)
        loss_1 += wbce_loss(y_hat[:, 3, :, :], targets_mid_3, weights=weights)
        loss_1 += wbce_loss(y_hat[:, 4, :, :], targets_mid_3, weights=weights)
        loss_2 =  wbce_loss(y_fuse, targets_mid_3, weights=weights)
        loss_3 =  F.mse_loss(y_r_hat, targets_coord)  

        loss = loss_1 + loss_2 + loss_3

    elif 'our' in model_name:
        connect_loss = ConnectLoss(length=400)
        reg_weight = cfg.loss_param.reg_weight
        connect_weight = cfg.loss_param.connect_weight / reg_weight
        prob_map, curves_limits = pred
        curves, limits = curves_limits[:, 0], curves_limits[:, 1]
        if 'carnet' in model_name:
            prob_map_1, prob_map_2 = prob_map
            loss_11 = wbce_loss(prob_map_1, targets_mid_3, weights=wce_weight)
            loss_12 = wbce_loss(prob_map_2, targets_mid_3, weights=wce_weight)
            loss_1 = loss_12 #+ loss_12
        else:
            loss_1 = wbce_loss(prob_map, targets_mid_3, weights=wce_weight)

        limits_mask = ~torch.isnan(targets_coord)
        loss_2 = F.binary_cross_entropy_with_logits(limits, limits_mask.to(dtype=limits.dtype))
        loss_3 = F.l1_loss(curves[limits_mask], targets_coord[limits_mask])

        if with_connect:
            loss_3 += connect_weight * connect_loss(curves, limits_mask.to(dtype=limits.dtype)) 
       
        loss = loss_1 + loss_2 + reg_weight * loss_3

    elif model_name == 'mdnet':
        maxpool = nn.MaxPool2d(kernel_size=2)
        ds_weigth = [0.25, 0.25, 0.25, 0.25]
        loss = Variable(torch.Tensor([0]).float()).cuda()
        
        for j, prob_map in enumerate(pred):
            if j > 0:
                targets_mid_3 = maxpool(targets_mid_3.float()).long()
            class_weight = Variable(torch.FloatTensor(ce_weight)).cuda() # 这里正例比较少，因此权重要大一些
            wce_weight = class_weight[targets_mid_3.long()] # (3, 4)
            loss_1 = wbce_loss(prob_map, targets_mid_3, weights=wce_weight) + dice_sigmoid_loss(prob_map, targets_mid_3) 
            loss += ds_weigth[j]*loss_1
        loss_1 = loss
        loss_2 = loss
        loss_3 = loss


    return loss, loss_1, loss_2, loss_3


def get_connect_regular_matrix(n):
    assert n >= 2
    diag_matrix = torch.eye(n)
    bottom_left_matrix = torch.zeros(n, n)
    bottom_left_matrix[:-1, 1:] = -torch.eye(n-1, n-1)
    res_matrix = diag_matrix + bottom_left_matrix
    res_matrix[0, 0] = 0
    
    return res_matrix



class ConnectLoss(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.regluar_matrix = get_connect_regular_matrix(self.length).cuda()
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        product = torch.mm(x, self.regluar_matrix)
        diff = torch.abs(product) - 1
        loss = (self.relu(diff) * mask).sum()

        return loss
















