# Changelog:
# - More robust feature selection in fit(): if 'time' missing, use 'date' (handled in data_loader).
# - inverse_transform_depth: accepts both numpy arrays and torch tensors.
# - API remains unchanged, compatible with rest of repository.

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch

class ScalerManager:
    def __init__(self):
        self.full_scaler = MinMaxScaler()
        self.depth_scaler = MinMaxScaler()

    def fit(self, data_frame):
        # Fit full_scaler on ['time','longitude','latitude','depth'] if available
        needed = ['time', 'longitude', 'latitude', 'depth']
        missing = [c for c in needed if c not in data_frame.columns]
        if missing:
            raise KeyError(f"ScalerManager.fit expects columns {needed}, but missing {missing}. "
                           "Ensure 'time' is prepared from 'date' if necessary.")
        self.full_scaler.fit(data_frame[['time', 'longitude', 'latitude', 'depth']])
        self.depth_scaler.fit(data_frame[['depth']])

    def transform_all(self, data_frame):
        """
        Normalize columns ['time','longitude','latitude','depth'] using fitted scaler.
        """
        return self.full_scaler.transform(data_frame[['time', 'longitude', 'latitude', 'depth']])

    def inverse_transform_all(self, norm_array):
        """
        Inverse scale normalized feature array back to physical values.
        """
        return self.full_scaler.inverse_transform(norm_array)

    def transform_depth(self, z_array):
        """
        Normalize 1D depth array.
        """
        z_np = np.array(z_array).reshape(-1, 1)
        return self.depth_scaler.transform(z_np)

    def inverse_transform_depth(self, z_norm_array):
        """
        Inverse normalize depth; accepts either numpy array or torch tensor.
        """
        if isinstance(z_norm_array, torch.Tensor):
            z_np = z_norm_array.detach().cpu().numpy().reshape(-1, 1)
        else:
            z_np = np.array(z_norm_array).reshape(-1, 1)
        return self.depth_scaler.inverse_transform(z_np)

    def inverse_x(self, x_norm_tensor):
        """
        Given normalized x, inverts to physical value.
        """
        x_np = x_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        dummy = np.zeros((x_np.shape[0], 4))
        dummy[:, 1] = x_np[:, 0]  # x is column 1
        return self.full_scaler.inverse_transform(dummy)[:, 1]

    def inverse_y(self, y_norm_tensor):
        """
        Given normalized y, inverts to physical value.
        """
        y_np = y_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        dummy = np.zeros((y_np.shape[0], 4))
        dummy[:, 2] = y_np[:, 0]  # y is column 2
        return self.full_scaler.inverse_transform(dummy)[:, 2]

    def inverse_time(self, t_norm_tensor):
        """
        Given normalized time, inverts to physical value.
        """
        t_np = t_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        dummy = np.zeros((t_np.shape[0], 4))
        dummy[:, 0] = t_np[:, 0]  # time is column 0
        return self.full_scaler.inverse_transform(dummy)[:, 0]

    def inverse_velocity(self, vel_norm_tensor):
        """
        By default, assumes velocity was not scaled. For custom scaling, override as needed.
        """
        vel_np = vel_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        return vel_np  # Replace with custom scaler if necessary