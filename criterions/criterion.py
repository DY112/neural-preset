import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
from itertools import combinations

class MasterCriterion(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.key_pairs = cfg.criterion.key_pairs
        
        self.mod_dict = {}
        self.mod_dict['l1_loss'] = L1_loss(cfg)
        self.mod_dict['cross_entropy'] = Cross_entropy(cfg)

    def forward(self, pred_dict, gt_dict, phase):
        
        loss_dict = {}
        total_loss = 0
        
        for loss_key in self.key_pairs:
            mod_key = self.cfg.criterion[loss_key].mod
            alpha = self.cfg.criterion[loss_key].alpha
            
            loss = self.mod_dict[mod_key](pred_dict, gt_dict)
            loss_dict[f'{phase}-{loss_key}'] = loss
            total_loss += (alpha * loss)
        
        loss_dict[f'{phase}-total_loss'] = total_loss

        return loss_dict

class L1_loss(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.L1 = nn.L1Loss()

    def forward(self, pred_dict, gt_dict):
        loss = self.L1(pred_dict['y_hat'], gt_dict['y'])

        return loss
    
class Cross_entropy(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred_dict, gt_dict):
        loss = self.cross_entropy(pred_dict['y_hat'], gt_dict['y'])

        return loss