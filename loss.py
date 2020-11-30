import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn.functional as F
from metric import dice_value, dice_coeff

def DiceLoss(input, target, apply_log=False):
    if apply_log:
        loss = - torch.log(dice_coeff(input, target))
    else:
        loss = 1 - dice_coeff(input, target)
    return loss.mean()

def KLDivLoss(input, target, smooth=1e-6):
    input = torch.sigmoid(input)
    loss = (input * (input / (target+smooth) + smooth).log()).sum()
    return loss

def CosineLoss(input, target):
    input = torch.sigmoid(input)
    cos = nn.CosineSimilarity()
    loss = cos(input, target)
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = self.BCE(inputs, targets)
        else:
            BCE_loss = self.BCE(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/101429
def WeightedBCE(logit, truth, weight=0.25):
    logit = logit.view(-1)
    truth = truth.view(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    if 0:
        loss = loss.mean()
    if 1:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (weight*pos*loss/pos_weight + (1-weight)*neg*loss/neg_weight).sum()

    return loss

# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

# Courtesy: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155201
def criterion_margin_focal_binary_cross_entropy(logit, truth):
    # logit = torch.sigmoid(logit)
    weight_pos=2
    weight_neg=1
    gamma=2
    margin=0.2
    em = np.exp(margin)

    logit = logit.view(-1)
    truth = truth.view(-1)
    log_pos = -F.logsigmoid(logit)
    log_neg = -F.logsigmoid(-logit)

    log_prob = truth*log_pos + (1-truth)*log_neg
    prob = torch.exp(-log_prob)
    margin = torch.log(em +(1-em)*prob)

    weight = truth*weight_pos + (1-truth)*weight_neg
    loss = margin + weight*(1 - prob) ** gamma * log_prob
    # loss = loss.mean()
    return loss