import torch
import torch.nn as nn
import numpy as np
from siren import SIRENLayer

class EnhancedPhysicsInformedThermocline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 主网络：用于输出基础速度、Z0 等
        self.main_net = nn.Sequential(
            SIRENLayer(4, 128, omega_0=30.0, is_first_layer=True),
            SIRENLayer(128, 128, omega_0=30.0),
            SIRENLayer(128, 128, omega_0=30.0),
            nn.Linear(128, 12)
        )

        # 可训练偏移项和 Ro
        self.B0_scalar = nn.Parameter(torch.tensor(0.01))
        self.B1_scalar = nn.Parameter(torch.tensor(0.01))
        self.Ro = nn.Parameter(torch.tensor(config.Ro))
        self.f = nn.Parameter(torch.tensor(1e-4), requires_grad=True)

        # ✅ 新增：theta_net 和 eta_net
        self.theta_net = nn.Sequential(
            SIRENLayer(4, 64, omega_0=30.0, is_first_layer=True),
            SIRENLayer(64, 64, omega_0=30.0),
            SIRENLayer(64, 1, omega_0=30.0)
        )

        self.eta_net = nn.Sequential(
            SIRENLayer(4, 64, omega_0=30.0, is_first_layer=True),
            SIRENLayer(64, 64, omega_0=30.0),
            SIRENLayer(64, 1, omega_0=30.0)
        )

    def forward(self, t, x, y, z):
        inputs = torch.cat([t, x, y, z], dim=1)
        return self.main_net(inputs)

    def compute_approximate_velocity(self, x, y, z, t):
        output = self.forward(t, x, y, z)
        u_bar = output[:, 0:1]
        v_bar = output[:, 1:2]
        Z0 = output[:, 9:10]

        # ✅ 构造 theta_raw 和 eta_raw
        theta_input = torch.cat([t, x, y, z], dim=1)
        eta_input = torch.cat([t, x, y, z], dim=1)
        theta_raw = self.theta_net(theta_input)
        eta_raw = self.eta_net(eta_input)

        # ✅ 构造指数衰减项
        decay = torch.exp(-torch.abs(z - Z0) / self.Ro)

        # ✅ 构造最终 theta 和 eta
        theta = theta_raw * decay
        eta = eta_raw * decay

        # 构造偏移项
        B0 = self.B0_scalar * torch.ones_like(u_bar)
        B1 = self.B1_scalar * torch.ones_like(v_bar)

        # 构造近似速度
        u_appr = u_bar + self.Ro * theta * u_bar + self.Ro ** 2 * eta * u_bar + B0
        v_appr = v_bar + self.Ro * theta * v_bar + self.Ro ** 2 * eta * v_bar + B1

        return u_appr, v_appr, u_bar, v_bar, theta, eta, Z0

    def compute_Z0(self, t, x, y):
        """
        计算 Z₀(t, x, y) 地形场
        输入必须是归一化后的 t, x, y
        """
        return self.dynamic_z0_net(t, x, y)

    def compute_base_velocity(self, x, y, z, t):
        output = self.forward(t, x, y, z)
        return output[:, 0:1], output[:, 1:2]

    def data_constraints(self, u_appr, v_appr, u_true, v_true):
        u_appr_unit = u_appr / (u_appr.norm(dim=1, keepdim=True) + 1e-6)
        v_appr_unit = v_appr / (v_appr.norm(dim=1, keepdim=True) + 1e-6)
        u_true_unit = u_true / (u_true.norm(dim=1, keepdim=True) + 1e-6)
        v_true_unit = v_true / (v_true.norm(dim=1, keepdim=True) + 1e-6)

        direction_loss = 1 - (u_appr_unit * u_true_unit + v_appr_unit * v_true_unit).sum(dim=1).mean()
        value_loss = torch.mean((u_appr - u_true)**2 + (v_appr - v_true)**2)
        return direction_loss, value_loss

    def rescale_to_physical_units(self, u_pred, v_pred):
        scale = self.config.velocity_scale
        return u_pred * scale, v_pred * scale

    def compute_physics_residual(self, x_norm, y_norm, depth_norm, t_norm):
        x_norm.requires_grad_(True)
        y_norm.requires_grad_(True)
        t_norm.requires_grad_(True)

        inputs = torch.cat([x_norm, y_norm, depth_norm, t_norm], dim=1)
        base_flow = self.forward(t_norm, x_norm, y_norm, depth_norm)
        u_bar = base_flow[:, 0:1]
        v_bar = base_flow[:, 1:2]
        g1 = base_flow[:, 5:6]
        g2 = base_flow[:, 6:7]

        def safe_grad(y, x):
            grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                       create_graph=True, retain_graph=True)[0]
            return torch.clamp(grad, -self.config.grad_clip, self.config.grad_clip)

        u_t = safe_grad(u_bar, t_norm)
        v_t = safe_grad(v_bar, t_norm)
        u_x = safe_grad(u_bar, x_norm)
        u_y = safe_grad(u_bar, y_norm)
        v_x = safe_grad(v_bar, x_norm)
        v_y = safe_grad(v_bar, y_norm)

        continuity = u_x + v_y

        M11 = g1 ** 2
        M12 = g1 * g2
        M21 = g1 * g2
        M22 = g2 ** 2

        f = getattr(self, "f", 1e-4)
        coriolis_u = -f * v_bar
        coriolis_v = f * u_bar

        P_x = torch.zeros_like(u_bar)
        P_y = torch.zeros_like(v_bar)
        h1 = torch.zeros_like(u_bar)
        h2 = torch.zeros_like(v_bar)

        res_u = u_t + u_bar * u_x + v_bar * u_y + self.config.Ro * u_bar \
                + M11 * u_bar + M12 * v_bar + coriolis_u + P_x - h1
        res_v = v_t + u_bar * v_x + v_bar * v_y + self.config.Ro * v_bar \
                + M21 * u_bar + M22 * v_bar + coriolis_v + P_y - h2

        return res_u, res_v, continuity

    def get_all_parameters(self):
        return {
            name: param.detach().cpu().numpy().tolist()
            for name, param in self.named_parameters()
        }
