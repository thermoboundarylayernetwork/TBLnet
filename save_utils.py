import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

from config import ModelConfig
from scaler_manager import ScalerManager

def save_model(model, output_dir, filename="trained_model.pt"):
    model_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

# Visualization and report saving
def save_visual_report(x, y, u_pred, v_pred, output_dir, Z0=None, t=None):
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    u_np = u_pred.detach().cpu().numpy().flatten()
    v_np = v_pred.detach().cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(x_np, y_np, u_np, v_np, scale=50, width=0.002, color="blue")
    ax.set_title("Velocity Field (u, v)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "velocity_field.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Velocity field plot saved to: {fig_path}")

    data = {
        "x": x_np,
        "y": y_np,
        "u_pred": u_np,
        "v_pred": v_np
    }
    if Z0 is not None:
        data["Z0"] = Z0.detach().cpu().numpy().flatten()
    if t is not None:
        data["t"] = t.detach().cpu().numpy().flatten()

    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "velocity_field.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Velocity field data saved to: {csv_path}")

class PredictionSaver:
    """
    Handles saving of predictions, diagnostic reports, and physical field export.
    All text files written with utf-8 encoding for maximum compatibility across platforms.
    """
    def __init__(self, model, scaler_mgr, output_dir):
        self.model = model
        self.scaler_mgr = scaler_mgr
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def inverse_all(self, x, y, z, t, u_pred, v_pred, u_true, v_true, Z0):
        return {
            "x": self.scaler_mgr.inverse_x(x).reshape(-1),
            "y": self.scaler_mgr.inverse_y(y).reshape(-1),
            "z": self.scaler_mgr.inverse_transform_depth(z).reshape(-1),
            "t": self.scaler_mgr.inverse_time(t).reshape(-1),
            "u_pred": self.scaler_mgr.inverse_velocity(u_pred).reshape(-1),
            "v_pred": self.scaler_mgr.inverse_velocity(v_pred).reshape(-1),
            "u_true": self.scaler_mgr.inverse_velocity(u_true).reshape(-1) if u_true is not None else None,
            "v_true": self.scaler_mgr.inverse_velocity(v_true).reshape(-1) if v_true is not None else None,
            "Z0": self.scaler_mgr.inverse_transform_depth(Z0).reshape(-1),
        }

    def save_Z0(self, x, y, z=None, t=None, u_pred=None, v_pred=None, Z0=None):
        if Z0 is None:
            raise ValueError("Z0 must be provided.")
        data = self.inverse_all(x, y, z, t, u_pred, v_pred, None, None, Z0)
        df = pd.DataFrame({k: v for k, v in data.items() if v is not None})
        path = os.path.join(self.output_dir, "Z0_prediction.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"Z₀ saved to: {path}")

    def save_full_prediction(self, original_df, u_pred, v_pred, u_true, v_true, Z0, x, y, z, t):
        import datetime
        self.model.eval()
        with torch.no_grad():
            data = self.inverse_all(x, y, z, t, u_pred, v_pred, u_true, v_true, Z0)
        df = pd.DataFrame(data)
        for col in ["time", "segment_id", "latitude", "longitude", "depth", "so", "thetao", "uo", "vo"]:
            if col in original_df.columns:
                df[col] = original_df[col].values[:len(df)]
        # Add human-readable time string column (base: 2023-03-01 00:00:00)
        t0 = datetime.datetime(2023, 3, 1, 0, 0, 0)
        if "time" in df.columns:
            df["date_str"] = df["time"].apply(
                lambda s: (t0 + datetime.timedelta(seconds=s)).strftime("%Y-%m-%d %H:%M:%S")
            )
        path = os.path.join(self.output_dir, "full_prediction.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"Full prediction saved to: {path}")

    # ---------- Directional & Gradient Consistency (safe for missing gradients) ----------

    @staticmethod
    def _safe_grad(output, inp):
        g = torch.autograd.grad(
            outputs=output,
            inputs=inp,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if g is None:
            g = torch.zeros_like(output)
        return g

    def compute_direction_metrics(self, u_pred, v_pred, u_true, v_true, g1, g2, Z0, x, y):
        x.requires_grad_(True)
        y.requires_grad_(True)
        Z0_x = self._safe_grad(Z0, x)
        Z0_y = self._safe_grad(Z0, y)
        g_vec = torch.stack([g1.flatten(), g2.flatten()], dim=1)
        Z0_grad = torch.stack([Z0_x.flatten(), Z0_y.flatten()], dim=1)
        g_unit = g_vec / (g_vec.norm(dim=1, keepdim=True) + 1e-8)
        Z0_unit = Z0_grad / (Z0_grad.norm(dim=1, keepdim=True) + 1e-8)
        dot_gz = (g_unit * Z0_unit).sum(dim=1)
        u_vec_true = torch.stack([u_true.flatten(), v_true.flatten()], dim=1)
        u_vec_pred = torch.stack([u_pred.flatten(), v_pred.flatten()], dim=1)
        u_unit_true = u_vec_true / (u_vec_true.norm(dim=1, keepdim=True) + 1e-8)
        u_unit_pred = u_vec_pred / (u_vec_pred.norm(dim=1, keepdim=True) + 1e-8)
        dot_uv = (u_unit_true * u_unit_pred).sum(dim=1)
        return dot_gz, dot_uv

    def compute_g_dot_gradZ0(self, g1, g2, Z0, x, y):
        x.requires_grad_(True)
        y.requires_grad_(True)
        Z0_x = self._safe_grad(Z0, x)
        Z0_y = self._safe_grad(Z0, y)
        g_vec = torch.stack([g1.flatten(), g2.flatten()], dim=1)
        Z0_grad = torch.stack([Z0_x.flatten(), Z0_y.flatten()], dim=1)
        g_unit = g_vec / (g_vec.norm(dim=1, keepdim=True) + 1e-8)
        Z0_unit = Z0_grad / (Z0_grad.norm(dim=1, keepdim=True) + 1e-8)
        return (g_unit * Z0_unit).sum(dim=1)

    # ---------- Reports and Diagnostics (utf-8, for cross-platform) ----------

    def save_direction_consistency(self, u_pred, v_pred, u_true, v_true, dot_gz, dot_uv):
        path = os.path.join(self.output_dir, "direction_consistency.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Velocity prediction errors:\n")
            f.write(f"u MAE: {(u_pred - u_true).abs().mean().item():.6f}\n")
            f.write(f"v MAE: {(v_pred - v_true).abs().mean().item():.6f}\n\n")
            f.write("g · ∇Z₀:\n")
            f.write(f"mean: {dot_gz.mean().item():.6f}, std: {dot_gz.std().item():.6f}\n\n")
            f.write("u_pred vs u_true (cosine similarity):\n")
            f.write(f"mean: {dot_uv.mean().item():.6f}, std: {dot_uv.std().item():.6f}\n")
        print(f"Direction consistency report saved to: {path}")

    def save_diagnostics(self, final_losses, g_dot_gradZ0=None, cos_sim=None, extra_metrics=None):
        path = os.path.join(self.output_dir, "diagnostic_metrics.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Model parameters:\n")
            f.write(f"Ro: {self.model.Ro.item():.6f}\n")
            f.write(f"B0_scalar: {self.model.B0_scalar.item():.6f}\n")
            f.write(f"B1_scalar: {self.model.B1_scalar.item():.6f}\n\n")

            if g_dot_gradZ0 is not None:
                f.write("g · ∇Z₀:\n")
                f.write(f"mean: {g_dot_gradZ0.mean().item():.6f}, std: {g_dot_gradZ0.std().item():.6f}\n\n")

            if cos_sim is not None:
                f.write("Predicted vs True Velocity Direction:\n")
                f.write(f"mean: {cos_sim.mean().item():.6f}, std: {cos_sim.std().item():.6f}\n\n")

            f.write("Loss terms:\n")
            for name, value in final_losses.items():
                if hasattr(value, "item"):
                    f.write(f"{name}: {value.item():.6f}\n")
                else:
                    f.write(f"{name}: {float(value):.6f}\n")

            if extra_metrics:
                f.write("\nAdditional metrics:\n")
                for name, value in extra_metrics.items():
                    f.write(f"{name}: {float(value):.6f}\n")
        print(f"Diagnostic metrics saved to: {path}")

    def compute_cos_similarity(self, u_pred, v_pred, u_true, v_true):
        u_vec_true = torch.stack([u_true.flatten(), v_true.flatten()], dim=1)
        u_vec_pred = torch.stack([u_pred.flatten(), v_pred.flatten()], dim=1)
        u_unit_true = u_vec_true / (u_vec_true.norm(dim=1, keepdim=True) + 1e-8)
        u_unit_pred = u_vec_pred / (u_vec_pred.norm(dim=1, keepdim=True) + 1e-8)
        return (u_unit_true * u_unit_pred).sum(dim=1)

    def plot_uv_diagnostics(self, u_pred, u_true, v_pred, v_true):
        u_pred_np = u_pred.detach().cpu().numpy().flatten()
        u_true_np = u_true.detach().cpu().numpy().flatten()
        v_pred_np = v_pred.detach().cpu().numpy().flatten()
        v_true_np = v_true.detach().cpu().numpy().flatten()
        u_res = u_pred_np - u_true_np
        v_res = v_pred_np - v_true_np

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].scatter(u_true_np, u_pred_np, alpha=0.3)
        axs[0, 0].plot(u_true_np, u_true_np, "r--")
        axs[0, 0].set_title("u: Predicted vs True")
        axs[0, 0].set_xlabel("True u")
        axs[0, 0].set_ylabel("Predicted u")

        axs[0, 1].scatter(v_true_np, v_pred_np, alpha=0.3)
        axs[0, 1].plot(v_true_np, v_true_np, "r--")
        axs[0, 1].set_title("v: Predicted vs True")
        axs[0, 1].set_xlabel("True v")
        axs[0, 1].set_ylabel("Predicted v")

        axs[1, 0].hist(u_res, bins=50, color="blue", alpha=0.7)
        axs[1, 0].set_title("u: Residual Histogram")
        axs[1, 0].set_xlabel("u_pred - u_true")

        axs[1, 1].hist(v_res, bins=50, color="green", alpha=0.7)
        axs[1, 1].set_title("v: Residual Histogram")
        axs[1, 1].set_xlabel("v_pred - v_true")

        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "uv_diagnostics.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Diagnostic plot saved to: {fig_path}")

    def plot_residual_histogram(self, u_pred, u_true, v_pred, v_true):
        u_res = (u_pred - u_true).detach().cpu().numpy().flatten()
        v_res = (v_pred - v_true).detach().cpu().numpy().flatten()
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].hist(u_res, bins=50, color="blue", alpha=0.7)
        axs[0].set_title("u: Residual Histogram")
        axs[0].set_xlabel("u_pred - u_true")
        axs[1].hist(v_res, bins=50, color="green", alpha=0.7)
        axs[1].set_title("v: Residual Histogram")
        axs[1].set_xlabel("v_pred - v_true")
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "uv_residual_histogram.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Residual histogram saved to: {fig_path}")

    def save_velocity_field(self, x, y, u_pred, v_pred, Z0=None, t=None):
        x_np = x.detach().cpu().numpy().flatten()
        y_np = y.detach().cpu().numpy().flatten()
        u_np = u_pred.detach().cpu().numpy().flatten()
        v_np = v_pred.detach().cpu().numpy().flatten()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.quiver(x_np, y_np, u_np, v_np, scale=50, width=0.002, color="blue")
        ax.set_title("Velocity Field (u, v)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, "velocity_field.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Velocity field plot saved to: {fig_path}")

        data = {
            "x": x_np,
            "y": y_np,
            "u_pred": u_np,
            "v_pred": v_np,
        }
        if Z0 is not None:
            data["Z0"] = Z0.detach().cpu().numpy().flatten()
        if t is not None:
            data["t"] = t.detach().cpu().numpy().flatten()

        df = pd.DataFrame(data)
        csv_path = os.path.join(self.output_dir, "velocity_field.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Velocity field data saved to: {csv_path}")

    def save_physics_fields(self, x, y, z, t,
                            Z0, theta, eta,
                            P, g1, g2, h1, h2,
                            time_phase_C2, time_phase_C3):
        """
        Save pointwise physical field data to CSV, including (optionally) spatial gradients for Z0.
        """
        x.requires_grad_(True)
        y.requires_grad_(True)

        Z0_x, Z0_y = torch.autograd.grad(
            outputs=Z0,
            inputs=(x, y),
            grad_outputs=torch.ones_like(Z0),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        if Z0_x is None:
            Z0_x = torch.zeros_like(Z0)
        if Z0_y is None:
            Z0_y = torch.zeros_like(Z0)

        Z0_grad_norm = torch.sqrt(Z0_x.flatten()**2 + Z0_y.flatten()**2 + 1e-12)

        df = pd.DataFrame({
            "x": self.scaler_mgr.inverse_x(x).reshape(-1),
            "y": self.scaler_mgr.inverse_y(y).reshape(-1),
            "z": self.scaler_mgr.inverse_transform_depth(z).reshape(-1),
            "t": self.scaler_mgr.inverse_time(t).reshape(-1),
            "Z0": self.scaler_mgr.inverse_transform_depth(Z0).reshape(-1),
            "theta": theta.detach().cpu().numpy().reshape(-1),
            "eta": eta.detach().cpu().numpy().reshape(-1),
            "P": P.detach().cpu().numpy().reshape(-1),
            "g1": g1.detach().cpu().numpy().reshape(-1),
            "g2": g2.detach().cpu().numpy().reshape(-1),
            "h1": h1.detach().cpu().numpy().reshape(-1),
            "h2": h2.detach().cpu().numpy().reshape(-1),
            "time_phase_C2": time_phase_C2.detach().cpu().numpy().reshape(-1),
            "time_phase_C3": time_phase_C3.detach().cpu().numpy().reshape(-1),
            "Z0_x": Z0_x.detach().cpu().numpy().reshape(-1),
            "Z0_y": Z0_y.detach().cpu().numpy().reshape(-1),
            "Z0_grad_norm": Z0_grad_norm.detach().cpu().numpy().reshape(-1),
        })

        path = os.path.join(self.output_dir, "physics_fields.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"Physical field data saved to: {path}")

    def save_physics_scalars(self):
        """Save global/scalar physical model parameters to JSON."""
        scalars = {
            "Ro": float(self.model.Ro.item()) if hasattr(self.model, "Ro") else None,
            "B0": float(self.model.B0_scalar.item()) if hasattr(self.model, "B0_scalar") else None,
            "B1": float(self.model.B1_scalar.item()) if hasattr(self.model, "B1_scalar") else None,
            "f": float(self.model.f.item()) if hasattr(self.model, "f") else None,
        }
        path = os.path.join(self.output_dir, "physics_scalars.json")
        with open(path, "w", encoding="utf-8") as fobj:
            json.dump(scalars, fobj, ensure_ascii=False, indent=2)
        print(f"Physical scalar parameters saved to: {path}")

    def save_all(self, original_df, u_pred, v_pred, u_true, v_true,
                 Z0, x, y, z, t, g1, g2,
                 final_losses, extra_metrics=None):
        """
        Save all results: Z0, full_prediction, velocity field, diagnostics, and plots.
        """
        self.save_Z0(x, y, z=z, t=t, u_pred=u_pred, v_pred=v_pred, Z0=Z0)
        self.save_full_prediction(original_df, u_pred, v_pred, u_true, v_true, Z0, x, y, z, t)
        self.save_velocity_field(x, y, u_pred, v_pred, Z0=Z0, t=t)
        # Directional consistency and diagnostics
        dot_gz, dot_uv = self.compute_direction_metrics(u_pred, v_pred, u_true, v_true, g1, g2, Z0, x, y)
        self.save_direction_consistency(u_pred, v_pred, u_true, v_true, dot_gz, dot_uv)
        self.save_diagnostics(final_losses, g_dot_gradZ0=dot_gz, cos_sim=dot_uv, extra_metrics=extra_metrics)
        self.plot_uv_diagnostics(u_pred, u_true, v_pred, v_true)
        print("All outputs saved.")