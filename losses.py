import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pandas as pd

from config import ModelConfig
from data_loader import load_csv_data
from physics_model import EnhancedPhysicsInformedThermocline
from compute_approximate_velocity import compute_velocity
from compute_physics_residual import compute_residuals
from physics_residual import geometric_constraint
from utils import get_device, LossManager, prepare_inputs
from save_utils import save_model, save_visual_report, PredictionSaver





class FixedDirectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, u_pred, v_pred, u_true, v_true):
        pred_velocity = torch.cat([u_pred, v_pred], dim=1)
        true_velocity = torch.cat([u_true, v_true], dim=1)

        pred_norm = torch.sqrt(torch.sum(pred_velocity ** 2, dim=1, keepdim=True) + self.epsilon)
        true_norm = torch.sqrt(torch.sum(true_velocity ** 2, dim=1, keepdim=True) + self.epsilon)

        pred_unit = pred_velocity / pred_norm
        true_unit = true_velocity / true_norm

        cosine_sim = torch.sum(pred_unit * true_unit, dim=1)
        cosine_sim = torch.clamp(cosine_sim, -0.999, 0.999)

        angle_rad = torch.acos(cosine_sim)
        angle_loss = angle_rad.mean()
        cosine_loss = (1 - cosine_sim).mean()

        final_loss = 0.7 * angle_loss + 0.3 * cosine_loss
        return final_loss







def direction_loss(u_pred, v_pred, u_true, v_true, min_magnitude=0.01, eps=1e-8):
    # 计算真实速度模长
    true_mag = torch.sqrt(u_true ** 2 + v_true ** 2)
    mask = (true_mag > 0.005)  # 仅在真实速度大的区域计算

    if not mask.any():
        return torch.tensor(0.0, device=u_pred.device, requires_grad=True)

    # 归一化预测向量
    pred_norm = torch.sqrt(u_pred ** 2 + v_pred ** 2 + eps)
    u_pred_n = u_pred / pred_norm
    v_pred_n = v_pred / pred_norm

    # 归一化真实向量
    true_norm = torch.sqrt(u_true ** 2 + v_true ** 2 + eps)
    u_true_n = u_true / true_norm
    v_true_n = v_true / true_norm

    # 余弦相似度
    cos_sim = u_pred_n * u_true_n + v_pred_n * v_true_n
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    return (1.0 - cos_sim[mask]).mean()


def value_loss(u_pred, v_pred, u_true, v_true):
    """
    数值损失：预测值与真实值的均方误差
    """
    true_mag = torch.sqrt(u_true ** 2 + v_true ** 2)
    weight_small = 0.9 * (true_mag <= 0.005) + 0.1 * (true_mag > 0.005)
    return torch.mean( ( (u_pred - u_true)**2 + (v_pred - v_true)**2 ) * weight_small )


def activation_loss(theta, eta):
    """
    激活项损失：鼓励 theta 和 eta 的输出幅度，避免被压制为零
    """
    theta_abs = torch.mean(torch.abs(theta))
    eta_abs = torch.mean(torch.abs(eta))

    # 放宽约束：鼓励输出幅度大于阈值
    loss_theta = F.relu(0.05 - theta_abs) ** 2
    loss_eta = F.relu(0.03 - eta_abs) ** 2

    # 控制结构比例：theta 不应远小于 eta
    #loss_ratio = F.relu(3 * eta_abs - theta_abs) ** 2

    return loss_theta + loss_eta #+ 1.0 * loss_ratio



# 完整的训练函数，包含早停机制


