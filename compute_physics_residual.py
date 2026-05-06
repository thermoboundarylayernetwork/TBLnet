def compute_physics_residual(self, x_norm, y_norm, depth_norm, t_norm):
    x_norm.requires_grad_(True)
    y_norm.requires_grad_(True)
    t_norm.requires_grad_(True)

    inputs = torch.cat([x_norm, y_norm, depth_norm, t_norm], dim=1)
    base_flow = self.base_flow_predictor(inputs)
    u_bar = base_flow[:, 0:1]
    v_bar = base_flow[:, 1:2]

    # Optionally: g1, g2 from network output
    g1 = base_flow[:, 5:6]  # Assume 6th column is g1
    g2 = base_flow[:, 6:7]  # 7th column is g2

    # Gradient computation with clipping
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

    # Continuity residual
    continuity = u_x + v_y

    # Construct g ⊗ g terms
    M11 = g1 ** 2
    M12 = g1 * g2
    M21 = g1 * g2
    M22 = g2 ** 2

    # Coriolis term (can be learnable or region-dependent)
    f = getattr(self, "f", 1e-4)  # Use self.f if defined, else default 1e-4
    coriolis_u = -f * v_bar
    coriolis_v = f * u_bar

    # Pressure gradient term (can be from network output or external source)
    P_x = torch.zeros_like(u_bar)  # Replaceable with base_flow[:, 8:9]
    P_y = torch.zeros_like(v_bar)

    # Forcing term (from network output or constant)
    h1 = torch.zeros_like(u_bar)
    h2 = torch.zeros_like(v_bar)

    # Momentum residuals
    res_u = u_t + u_bar * u_x + v_bar * u_y + self.config.Ro * u_bar + M11 * u_bar + M12 * v_bar + coriolis_u + P_x - h1
    res_v = v_t + u_bar * v_x + v_bar * v_y + self.config.Ro * v_bar + M21 * u_bar + M22 * v_bar + coriolis_v + P_y - h2

    return res_u, res_v, continuity

def compute_residuals(model, x, y, z, t):
    """
    Wrapper function: computes physics residuals given normalized input tensors.
    Returns momentum and continuity residuals.
    """
    return model.compute_physics_residual(x, y, z, t)