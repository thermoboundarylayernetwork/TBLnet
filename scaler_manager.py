# 修改说明：
# - 对 fit 中选列做更健壮的检查：如果 'time' 不存在但有 'date'，优先使用 'time'（data_loader 已创建）
# - inverse_transform_depth: 增强兼容性（支持 numpy 或 tensor）
# - 保持接口不变，兼容仓库其余代码

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch

class ScalerManager:
    def __init__(self):
        self.full_scaler = MinMaxScaler()
        self.depth_scaler = MinMaxScaler()

    def fit(self, data_frame):
        # 尝试使用 'time','longitude','latitude','depth' 这四列来拟合 full_scaler
        needed = ['time', 'longitude', 'latitude', 'depth']
        missing = [c for c in needed if c not in data_frame.columns]
        if missing:
            raise KeyError(f"ScalerManager.fit expects columns {needed}, but missing {missing}. "
                           "Make sure data_loader created 'time' from 'date' if necessary.")
        self.full_scaler.fit(data_frame[['time', 'longitude', 'latitude', 'depth']])
        self.depth_scaler.fit(data_frame[['depth']])

    def transform_all(self, data_frame):
        return self.full_scaler.transform(data_frame[['time', 'longitude', 'latitude', 'depth']])

    def inverse_transform_all(self, norm_array):
        return self.full_scaler.inverse_transform(norm_array)

    def transform_depth(self, z_array):
        z_np = np.array(z_array).reshape(-1, 1)
        return self.depth_scaler.transform(z_np)

    def inverse_transform_depth(self, z_norm_array):
        # accept numpy array or torch tensor
        if isinstance(z_norm_array, torch.Tensor):
            z_np = z_norm_array.detach().cpu().numpy().reshape(-1, 1)
        else:
            z_np = np.array(z_norm_array).reshape(-1, 1)
        return self.depth_scaler.inverse_transform(z_np)

    def inverse_x(self, x_norm_tensor):
        x_np = x_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        # 构造一个虚拟输入，填充其他列
        dummy = np.zeros((x_np.shape[0], 4))
        dummy[:, 1] = x_np[:, 0]  # x 是第2列（索引1）
        return self.full_scaler.inverse_transform(dummy)[:, 1]  # 返回反变换后的 x

    def inverse_y(self, y_norm_tensor):
        y_np = y_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        dummy = np.zeros((y_np.shape[0], 4))
        dummy[:, 2] = y_np[:, 0]  # y 是第3列（索引2）
        return self.full_scaler.inverse_transform(dummy)[:, 2]  # 返回反变换后的 y

    def inverse_time(self, t_norm_tensor):
        t_np = t_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        dummy = np.zeros((t_np.shape[0], 4))
        dummy[:, 0] = t_np[:, 0]  # time 是第1列（索引0）
        return self.full_scaler.inverse_transform(dummy)[:, 0]

    def inverse_velocity(self, vel_norm_tensor):
        vel_np = vel_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        # 如果你对速度只做了缩放，可以直接反变换；否则需要额外 scaler
        return vel_np  # 或使用 self.velocity_scaler.inverse_transform(vel_np)