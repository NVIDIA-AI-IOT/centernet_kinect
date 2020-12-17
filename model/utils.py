'''
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import torch
import torchvision
import torch.nn.functional as F

# Adding Project Path
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(PROJ_PATH)

# Importing Project Libraries
import pipeline.constants as const

def Criterion(pred: torch.tensor, ground_truth: torch.tensor, loss_type=const.LOSS,\
                alpha=const.ALPHA, beta=const.BETA, num_classes=const.NUM_CLASSES,\
                lambda_h=const.LAMBDA_HEATMAP, lambda_s=const.LAMBDA_SIZE, lambda_o=const.LAMBDA_OFFSET):
    """
    Logistic Regression Loss from Centernet Paper

    :param pred: torch.tensor, model prediction (N, Num_classes+4, 160, 160)
    :param ground_truth: torch.tensor, ground truth (N, Num_classes+4, 160, 160)
    :param alpha: int, constant defined in the paper
    :param beta: int, constant defined in the paper
    """
    loss_switcher = {
        "MSE" : mse_loss,
        "Logistic": logisitc_reg_loss,
    }
    mas_loss = loss_switcher[loss_type]
    mask_loss = mas_loss(pred[:,0:num_classes], ground_truth[:,0:num_classes])
    # BBox Sizes
    size_loss_y = regression_loss(pred[:,-4], ground_truth[:,-4], ground_truth[:,0:num_classes])
    size_loss_x = regression_loss(pred[:,-3], ground_truth[:,-3], ground_truth[:,0:num_classes])
    # Offset Losses
    offset_loss_y = regression_loss(pred[:,-2], ground_truth[:,-2], ground_truth[:,0:num_classes])
    offset_loss_x = regression_loss(pred[:,-1], ground_truth[:,-1], ground_truth[:,0:num_classes])

    loss = lambda_h*mask_loss + lambda_s*(size_loss_x+size_loss_y) + lambda_o*(offset_loss_x+offset_loss_y)

    return loss

class Summary(object):
    """
    Tracking summary of a variable
    """
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, value):
        self.value = value
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count

############################
###  Internal Functions  ###
############################

def logisitc_reg_loss(pred: torch.tensor, ground_truth_heatmap: torch.tensor,\
                alpha=const.ALPHA, beta=const.BETA, thresh=const.THRESHOLD, window_size=11):
    """
    Logistic Regression Loss from Centernet Paper

    :param pred: torch.tensor, model prediction (N, Num_classes, 160, 160)
    :param ground_truth_heatmap: torch.tensor, ground truth (N, Num_classes, 160, 160)
    :param alpha: int, constant defined in the paper
    :param beta: int, constant defined in the paper
    :return: logistic regression loss (regressing heatmaps)
    """
    p = pred
    # p = torch.max_pool2d(pred, kernel_size=window_size, stride=1, padding=window_size//2)
    # p = ((pred == p) * (p > thresh)).float()
    p[p.lt(1e-3)] = 1e-3
    p[p.gt(.99)] = .99

    gt = ground_truth_heatmap

    # pos_inds = gt.eq(1).float()
    # neg_inds = gt.lt(1).float()

    pos_inds = gt.gt(thresh).float() +  gt.eq(thresh).float()
    neg_inds = gt.lt(thresh).float()

    # weights = gt[pos_inds.bool()].sum()
    weights = 1

    pos_loss = torch.log(p) * torch.pow(1-p, alpha) * pos_inds * weights
    neg_loss = torch.log(1-p) * torch.pow(p, alpha) * torch.pow(1-gt, beta) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.float().sum()
    neg_loss = neg_loss.float().sum()

    if torch.isnan(-1*(pos_loss + neg_loss)/num_pos if num_pos!=0 else -1*neg_loss):
        import pdb; pdb.set_trace()


    return -1*(pos_loss + neg_loss)/num_pos if num_pos!=0 else -1*neg_loss

def mse_loss(pred: torch.tensor, ground_truth_heatmap: torch.tensor):
    """
    Logistic Regression Loss from Centernet Paper

    :param pred: torch.tensor, model prediction (N, Num_classes, 160, 160)
    :param ground_truth_heatmap: torch.tensor, ground truth (N, Num_classes, 160, 160)
    :param alpha: int, constant defined in the paper
    :param beta: int, constant defined in the paper
    :return: logistic regression loss (regressing heatmaps)
    """

    loss = torch.nn.MSELoss()

    p = pred
    gt = ground_truth_heatmap

    return loss(p, gt)

def regression_loss(pred: torch.tensor, ground_truth: torch.tensor, ground_truth_heatmap: torch.tensor):
    """
    Regression Loss from Centernet Paper

    :param pred: torch.tensor, model prediction (N, 1, 160, 160)
    :param ground_truth: torch.tensor, ground truth (N, 1, 160, 160)
    :param ground_truth_heatmap: torch.tensor, ground truth (N, Num_classes, 160, 160)
    :return: regression loss (regression size and offsets)
    """

    num = ground_truth_heatmap.float().sum()*2
    
    mask, _ = ground_truth_heatmap.max(1)
    mask = mask.eq(1).float()

    pred = pred[mask==1]
    ground_truth = ground_truth[mask==1]
    
    reg_loss = F.l1_loss(
        pred, ground_truth, size_average=False
    )

    # reg_loss = reg_loss / (num + 1e-4)
    return reg_loss