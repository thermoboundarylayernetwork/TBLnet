def compute_approximate_velocity(self, x_norm, y_norm, depth_norm, t_norm):
    """
    Multi-scale asymptotic velocity prediction:
    u_appr ≈ ū + Ro·θ·ū + Ro²·η·ū + B0
    v_appr ≈ v̄ + Ro·θ·v̄ + Ro²·η·v̄ + B1
    """

    # Concatenate normalized input features
    inputs = torch.cat([x_norm, y_norm, depth_norm, t_norm], dim=1)

    # Base flow prediction
    base_flow = self.base_flow_predictor(inputs)
    u_bar = base_flow[:, 0:1]
    v_bar = base_flow[:, 1:2]

    # Dynamic thermocline depth Z₀(t, x, y)
    Z0 = self.dynamic_z0_net(t_norm, x_norm, y_norm)

    # Correction terms θ(t, x, y, z) and η(t, x, y, z)
    theta = self.depth_aware_theta_net(x_norm, y_norm, depth_norm, t_norm, Z0)
    eta = self.depth_aware_eta_net(x_norm, y_norm, depth_norm, t_norm, Z0)

    B0 = self.B0_scalar * torch.ones_like(u_bar)
    B1 = self.B1_scalar * torch.ones_like(v_bar)

    u_appr = u_bar + self.Ro * theta * u_bar + self.Ro ** 2 * eta * u_bar + B0
    v_appr = v_bar + self.Ro * theta * v_bar + self.Ro ** 2 * eta * v_bar + B1

    # Return predicted velocities and physical quantities
    return u_appr, v_appr, u_bar, v_bar, theta, eta, Z0


def compute_velocity(model, x, y, z, t):
    """
    Main interface: returns velocity predictions and physical quantities.
    """
    # Compute asymptotic velocity prediction
    u_appr, v_appr, u_bar, v_bar, theta, eta, Z0 = model.compute_approximate_velocity(x, y, z, t)

    # Forward pass for additional diagnostics
    output = model.forward(t, x, y, z)

    # Extract physical quantities
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