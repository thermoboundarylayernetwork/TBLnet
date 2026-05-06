def compute_physics_residual(self, x_norm, y_norm, depth_norm, t_norm):
    x_norm.requires_grad_(True)
    y_norm.requires_grad_(True)
    t_norm.requires_grad_(True)

    inputs = torch.cat([x_norm, y_norm, depth_norm, t_norm], dim=1)
    base_flow = self.base_flow_predictor(inputs)
    u_bar = base_flow[:, 0:1]
    v_bar = base_flow[:, 1:2]

    # 可选：g1, g2 从网络输出
    g1 = base_flow[:, 5:6]  # 假设第6列是 g1
    g2 = base_flow[:, 6:7]  # 第7列是 g2

    # 导数计算 + 裁剪
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

    # 连续性残差
    continuity = u_x + v_y

    # 构造 g ⊗ g 项
    M11 = g1 ** 2
    M12 = g1 * g2
    M21 = g1 * g2
    M22 = g2 ** 2

    # 科氏力项（可设为 learnable 或区域变异）
    f = getattr(self, "f", 1e-4)  # 如果你定义了 self.f 作为 nn.Parameter
    coriolis_u = -f * v_bar
    coriolis_v = f * u_bar

    # 压力梯度项（可从网络输出或外部提供）
    P_x = torch.zeros_like(u_bar)  # 可替换为 base_flow[:, 8:9]
    P_y = torch.zeros_like(v_bar)

    # 外力项（可从网络输出或设为常数）
    h1 = torch.zeros_like(u_bar)
    h2 = torch.zeros_like(v_bar)

    # 动量残差
    res_u = u_t + u_bar * u_x + v_bar * u_y + self.config.Ro * u_bar + M11 * u_bar + M12 * v_bar + coriolis_u + P_x - h1
    res_v = v_t + u_bar * v_x + v_bar * v_y + self.config.Ro * v_bar + M21 * u_bar + M22 * v_bar + coriolis_v + P_y - h2

    return res_u, res_v, continuity

def compute_residuals(model, x, y, z, t):
    """
    外部调用接口：包装模型的 compute_physics_residual 方法
    输入为标准化后的张量，输出为物理残差和相关变量
    """
    return model.compute_physics_residual(x, y, z, t)
