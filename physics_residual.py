import torch
import torch.nn as nn

class PhysicsResidualModule(nn.Module):
    """
    Module to compute physics-based residuals (momentum, continuity, and energy) for model outputs.
    Supports optional adaptive weighting for multi-term residuals.
    """
    def __init__(self, model, auto_weight=True):
        super().__init__()
        self.model = model
        self.auto_weight = auto_weight
        self.register_buffer("residual_stats", torch.zeros(3))  # [momentum, continuity, energy]

    def compute(self, x, y, z, t):
        for var in [x, y, z, t]:
            var.requires_grad_(True)

        # Model prediction
        (u_pred, v_pred, u_bar, v_bar, theta, eta, Z0,
         P, g1, g2, h1, h2, time_phase_C2, time_phase_C3) = self.model.compute_velocity(x, y, z, t)

        # Continuity residual (divergence of velocity)
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        v_y = torch.autograd.grad(v_pred, y, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
        res_cont = u_x + v_y

        # Momentum terms (u and v components)
        u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        v_t = torch.autograd.grad(v_pred, t, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
        u_y = torch.autograd.grad(u_pred, y, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        v_x = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]

        # Pressure gradient terms (optional)
        P_x = torch.autograd.grad(P, x, grad_outputs=torch.ones_like(P), create_graph=True)[0]
        P_y = torch.autograd.grad(P, y, grad_outputs=torch.ones_like(P), create_graph=True)[0]

        # Effective Coriolis parameter
        Ro = self.model.Ro
        epsilon = 1e-3
        f_eff = 1.0 / (Ro + epsilon)

        # Momentum residuals
        res_u = u_t + u_pred * u_x + v_pred * u_y - f_eff * v_pred + P_x - h1
        res_v = v_t + u_pred * v_x + v_pred * v_y + f_eff * u_pred + P_y - h2

        # Optional energy loss (mean kinetic energy)
        energy_loss = torch.mean(0.5 * (u_pred ** 2 + v_pred ** 2))

        # Update stats for adaptive weighting, if enabled
        if self.auto_weight:
            self.residual_stats[0] = res_u.abs().mean().detach()
            self.residual_stats[1] = res_cont.abs().mean().detach()
            self.residual_stats[2] = energy_loss.detach()

        return {
            "res_u": res_u,
            "res_v": res_v,
            "res_cont": res_cont,
            "energy_loss": energy_loss,
            "weights": self.get_weights() if self.auto_weight else None
        }

    def get_weights(self):
        """
        Returns normalized inverse-magnitude weights for residual terms.
        """
        inv = 1.0 / (self.residual_stats + 1e-6)
        return inv / inv.sum()

def geometric_constraint(g1, g2, Z0, x, y, min_magnitude=1e-3, weight_magnitude=0.1):
    """
    Penalize the misalignment and lack of magnitude between vector (g1, g2)
    and the gradient of Z0 with respect to x and y.
    """
    grad_Z0_x = torch.autograd.grad(Z0, x, grad_outputs=torch.ones_like(Z0), create_graph=True, retain_graph=True, allow_unused=True)[0]
    grad_Z0_y = torch.autograd.grad(Z0, y, grad_outputs=torch.ones_like(Z0), create_graph=True, retain_graph=True, allow_unused=True)[0]

    if grad_Z0_x is None or grad_Z0_y is None:
        raise ValueError("Z0 does not have gradient connection with respect to x or y.")

    grad_vec = torch.stack([grad_Z0_x, grad_Z0_y], dim=1)
    g_vec = torch.stack([g1, g2], dim=1)

    grad_unit = grad_vec / (torch.norm(grad_vec, dim=1, keepdim=True) + 1e-8)
    g_unit = g_vec / (torch.norm(g_vec, dim=1, keepdim=True) + 1e-8)

    direction_loss = 1.0 - torch.sum(grad_unit * g_unit, dim=1).mean()
    magnitude_loss = torch.mean(torch.relu(min_magnitude - torch.norm(g_vec, dim=1)) ** 2)

    return direction_loss + weight_magnitude * magnitude_loss