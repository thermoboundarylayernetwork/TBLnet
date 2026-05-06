#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for:
- Selecting the execution device (cuda/mps/cpu)
- Preparing inputs with requires_grad for autograd
- Optional residual consistency diagnostics
- Managing weighted multi-term losses in a consistent way
"""

import torch


def get_device():
    """
    Select an available device in the order: CUDA -> MPS -> CPU.
    """
    if torch.cuda.is_available():
        print("[INFO] Using device: cuda")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("[INFO] Using device: mps")
        return torch.device("mps")
    print("[INFO] Using device: cpu")
    return torch.device("cpu")


def rescale_to_physical_units(u_pred, v_pred, velocity_scale=1.0):
    """
    Convert predicted velocities from normalized units to physical units (e.g., m/s).
    """
    u_out = u_pred * velocity_scale
    v_out = v_pred * velocity_scale
    return u_out, v_out


def prepare_inputs(x, y, z, t):
    """
    Ensure input tensors have requires_grad=True to support autograd-based derivatives.
    Returns cloned, detached tensors with gradients enabled.
    """
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    return x, y, z, t


def validate_residual_consistency(u_true, u_pred, v_true, v_pred, output_dir):
    """
    Optional diagnostic:
    - Scatter plots (predicted vs true)
    - Residual histograms
    - A small text report containing residual mean, skewness, and bias ratio
    """
    # Local import to avoid forcing matplotlib as a hard dependency for non-plotting runs
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    def analyze(name, true, pred):
        true = true.detach().cpu().flatten()
        pred = pred.detach().cpu().flatten()
        residual = pred - true

        mean_residual = residual.mean().item()
        skewness = ((residual - residual.mean()) ** 3).mean() / (residual.std() ** 3 + 1e-8)
        bias_direction = (pred < true).float().mean().item()  # fraction of under-predictions

        # Scatter plot
        plt.figure(figsize=(5, 5))
        plt.scatter(true.numpy(), pred.numpy(), alpha=0.3, s=10)
        plt.plot([true.min(), true.max()], [true.min(), true.max()], "r--")
        plt.xlabel(f"{name}_true")
        plt.ylabel(f"{name}_pred")
        plt.title(f"{name}: Predicted vs True")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_scatter_check.png")
        plt.close()

        # Residual histogram
        plt.figure(figsize=(5, 4))
        plt.hist(residual.numpy(), bins=50, color="lightgreen", edgecolor="black")
        plt.title(f"{name} Residual Histogram")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_residual_hist_check.png")
        plt.close()

        return {
            "mean_residual": mean_residual,
            "skewness": float(skewness.item() if hasattr(skewness, "item") else skewness),
            "under_prediction_ratio": bias_direction,
        }

    u_report = analyze("u", u_true, u_pred)
    v_report = analyze("v", v_true, v_pred)

    report_path = f"{output_dir}/residual_consistency_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        for name, report in zip(["u", "v"], [u_report, v_report]):
            f.write(f"{name}:\n")
            for k, v in report.items():
                f.write(f"  {k}: {v:.6f}\n")
            f.write("\n")

    print(f"[INFO] Residual consistency report saved: {report_path}")


class LossManager:
    """
    A lightweight manager for weighted multi-term losses.

    Notes:
    - `weights` defines which loss keys are included in the total loss.
    - `compute_total_loss()` will sum only keys present in self.weights.
    - Call `set_weights()` to update weights during training (e.g., schedules).
    """

    def __init__(self, weights=None):
        self.weights = weights or {
            "data": 1.0,
            "dir": 1.0,
            "phys": 1.0,
            "cont": 1.0,
            "geo": 1.0,
            "act": 0.01,
            "var": 0.01,
        }
        self.history = []

    def compute_total_loss(self, losses: dict, epoch: int = None):
        # Only sum the loss terms that appear in weights
        total = sum(self.weights[key] * losses[key] for key in self.weights)

        # Store a scalar history snapshot for debugging/plotting
        snap = {"epoch": epoch, "total_loss": float(total.detach().cpu().item())}
        for k, v in losses.items():
            try:
                snap[f"{k}_loss"] = float(v.detach().cpu().item())
            except Exception:
                # Fallback: if v is not a tensor
                snap[f"{k}_loss"] = float(v)
        self.history.append(snap)

        return total

    def get_history(self):
        return self.history

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights: dict):
        self.weights.update(new_weights)