def compute_approximate_velocity(self, x_norm, y_norm, depth_norm, t_norm):
    """
    多尺度渐近速度预测：
    u_appr ≈ ū + Ro·θ·ū + Ro²·η·ū + B0
    v_appr ≈ v̄ + Ro·θ·v̄ + Ro²·η·v̄ + B1
    """

    # 🔹 拼接输入特征
    inputs = torch.cat([x_norm, y_norm, depth_norm, t_norm], dim=1)

    # 🔹 基础流场（增强版）
    base_flow = self.base_flow_predictor(inputs)
    u_bar = base_flow[:, 0:1]
    v_bar = base_flow[:, 1:2]

    # 🔹 动态温跃层深度 Z₀(t,x,y)
    Z0 = self.dynamic_z0_net(t_norm, x_norm, y_norm)

    # 🔹 修正项 θ(t,x,y,z) 和 η(t,x,y,z)
    theta = self.depth_aware_theta_net(x_norm, y_norm, depth_norm, t_norm, Z0)
    eta = self.depth_aware_eta_net(x_norm, y_norm, depth_norm, t_norm, Z0)

    B0 = self.B0_scalar * torch.ones_like(u_bar)
    B1 = self.B1_scalar * torch.ones_like(v_bar)

    u_appr = u_bar + self.Ro * theta * u_bar + self.Ro ** 2 * eta * u_bar + B0
    v_appr = v_bar + self.Ro * theta * v_bar + self.Ro ** 2 * eta * v_bar + B1


    # 🔹 返回基础预测和修正项（不再调用 self.forward）
    return u_appr, v_appr, u_bar, v_bar, theta, eta, Z0


def compute_velocity(model, x, y, z, t):
    """
    外部调用接口：返回速度预测 + 物理量
    """
    # 🔹 主速度近似部分
    u_appr, v_appr, u_bar, v_bar, theta, eta, Z0 = model.compute_approximate_velocity(x, y, z, t)

    # ✅ 修正参数顺序：t 在前
    output = model.forward(t, x, y, z)

    # 🔹 提取物理量
    P = output[:, 4:5]
    g1 = output[:, 5:6]
    g2 = output[:, 6:7]
    h1 = output[:, 7:8]
    h2 = output[:, 8:9]
    time_phase_C2 = output[:, 10:11]
    time_phase_C3 = output[:, 11:12]

    return (
        u_appr, v_appr, u_bar, v_bar, theta, eta, Z0,
        P, g1, g2, h1, h2, time_phase_C2, time_phase_C3
    )

