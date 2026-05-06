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
    print(f"模型已保存到: {model_path}")


# 可视化报告保存
def save_visual_report(x, y, u_pred, v_pred, output_dir, Z0=None, t=None):
    # 转为 numpy
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    u_np = u_pred.detach().cpu().numpy().flatten()
    v_np = v_pred.detach().cpu().numpy().flatten()

    # 保存速度场图（quiver）
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
    print(f"速度场图已保存到: {fig_path}")

    # 保存 CSV
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
    print(f"速度场数据已保存到: {csv_path}")


class PredictionSaver:
    """
    负责保存预测结果、诊断报告、以及物理量导出。
    修复点：
    - 所有文本写入统一使用 utf-8，避免 Windows 默认 gbk 报错
    - save_physics_scalars() 依赖 json：已在文件顶部 import json
    - compute_direction_metrics() / compute_g_dot_gradZ0() 对 autograd.grad 增加 allow_unused=True，
      并对 None 梯度回填 0，避免“Tensor not used in graph”导致保存阶段崩溃
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
        print(f"Z₀ 已保存到: {path}")

    def save_full_prediction(self, original_df, u_pred, v_pred, u_true, v_true, Z0, x, y, z, t):
        import datetime

        self.model.eval()
        with torch.no_grad():
            data = self.inverse_all(x, y, z, t, u_pred, v_pred, u_true, v_true, Z0)

        df = pd.DataFrame(data)

        # 合并原始数据列（如果存在）
        for col in ["time", "segment_id", "latitude", "longitude", "depth", "so", "thetao", "uo", "vo"]:
            if col in original_df.columns:
                df[col] = original_df[col].values[:len(df)]

        # 添加时间字符串列（假设起始时间为 2023-03-01 00:00:00）
        t0 = datetime.datetime(2023, 3, 1, 0, 0, 0)
        if "time" in df.columns:
            df["date_str"] = df["time"].apply(
                lambda s: (t0 + datetime.timedelta(seconds=s)).strftime("%Y-%m-%d %H:%M:%S")
            )

        path = os.path.join(self.output_dir, "full_prediction.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"完整预测结果已保存到: {path}")

    # ---------- 方向一致性/梯度相关（关键修复：allow_unused + None->0） ----------

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
        # 注意：必须在计算 Z0 之前就 requires_grad=True 才能保证图完整。
        # 这里仍然设一次，作为兜底；如果 Z0 的图里确实没用到 x/y，则 grad 会是 None -> 0（不会崩）。
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

    # ---------- 文本报告：统一 utf-8 ----------

    def save_direction_consistency(self, u_pred, v_pred, u_true, v_true, dot_gz, dot_uv):
        path = os.path.join(self.output_dir, "direction_consistency.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("速度预测误差:\n")
            f.write(f"u MAE: {(u_pred - u_true).abs().mean().item():.6f}\n")
            f.write(f"v MAE: {(v_pred - v_true).abs().mean().item():.6f}\n\n")
            f.write("g · ∇Z₀:\n")
            f.write(f"mean: {dot_gz.mean().item():.6f}, std: {dot_gz.std().item():.6f}\n\n")
            f.write("u_pred vs u_true:\n")
            f.write(f"mean: {dot_uv.mean().item():.6f}, std: {dot_uv.std().item():.6f}\n")
        print(f"方向一致性报告已保存到: {path}")

    def save_diagnostics(self, final_losses, g_dot_gradZ0=None, cos_sim=None, extra_metrics=None):
        path = os.path.join(self.output_dir, "diagnostic_metrics.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("模型参数:\n")
            f.write(f"Ro: {self.model.Ro.item():.6f}\n")
            f.write(f"B0_scalar: {self.model.B0_scalar.item():.6f}\n")
            f.write(f"B1_scalar: {self.model.B1_scalar.item():.6f}\n\n")

            if g_dot_gradZ0 is not None:
                f.write("g · ∇Z₀:\n")
                f.write(f"mean: {g_dot_gradZ0.mean().item():.6f}, std: {g_dot_gradZ0.std().item():.6f}\n\n")

            if cos_sim is not None:
                f.write("Predicted vs True Velocity Direction:\n")
                f.write(f"mean: {cos_sim.mean().item():.6f}, std: {cos_sim.std().item():.6f}\n\n")

            f.write("损失项:\n")
            for name, value in final_losses.items():
                # value 可能是 tensor 或 float
                if hasattr(value, "item"):
                    f.write(f"{name}: {value.item():.6f}\n")
                else:
                    f.write(f"{name}: {float(value):.6f}\n")

            if extra_metrics:
                f.write("\n其他指标:\n")
                for name, value in extra_metrics.items():
                    f.write(f"{name}: {float(value):.6f}\n")

        print(f"诊断指标已保存到: {path}")

    # ---------- 相似度/绘图 ----------

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
        print(f"综合诊断图已保存到: {fig_path}")

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
        print(f"残差直方图已保存到: {fig_path}")

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
        print(f"速度场图已保存到: {fig_path}")

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
        print(f"速度场数据已保存到: {csv_path}")

    # ---------- 物理场/标量导出 ----------

    def save_physics_fields(self, x, y, z, t,
                            Z0, theta, eta,
                            P, g1, g2, h1, h2,
                            time_phase_C2, time_phase_C3):
        """
        保存物理场相关的逐点输出到 CSV：
        包含：反归一化的 x,y,z,t、Z0（反归一化）、theta、eta、P、g1/g2、h1/h2、time_phase_C2/C3、
              以及 Z0 在 x、y 方向的梯度（Z0_x, Z0_y）与梯度范数（Z0_grad_norm）
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
        print(f"物理场逐点输出已保存到: {path}")

    def save_physics_scalars(self):
        """保存模型中学习到的全局/标量参数到 JSON"""
        scalars = {
            "Ro": float(self.model.Ro.item()) if hasattr(self.model, "Ro") else None,
            "B0": float(self.model.B0_scalar.item()) if hasattr(self.model, "B0_scalar") else None,
            "B1": float(self.model.B1_scalar.item()) if hasattr(self.model, "B1_scalar") else None,
            "f": float(self.model.f.item()) if hasattr(self.model, "f") else None,
        }
        path = os.path.join(self.output_dir, "physics_scalars.json")
        with open(path, "w", encoding="utf-8") as fobj:
            json.dump(scalars, fobj, ensure_ascii=False, indent=2)
        print(f"物理标量参数已保存到: {path}")

    # ---------- 总保存入口 ----------

    def save_all(self, original_df, u_pred, v_pred, u_true, v_true,
                 Z0, x, y, z, t, g1, g2,
                 final_losses, extra_metrics=None):
        # 保存 Z₀
        self.save_Z0(x, y, z=z, t=t, u_pred=u_pred, v_pred=v_pred, Z0=Z0)

        # 保存完整预测 CSV
        self.save_full_prediction(original_df, u_pred, v_pred, u_true, v_true, Z0, x, y, z, t)

        # 保存速度场图和 CSV
        self.save_velocity_field(x, y, u_pred, v_pred, Z0=Z0, t=t)

        # 统一计算方向一致性指标（关键：不会因梯度不存在而崩）
        dot_gz, dot_uv = self.compute_direction_metrics(u_pred, v_pred, u_true, v_true, g1, g2, Z0, x, y)

        # 保存方向一致性报告
        self.save_direction_consistency(u_pred, v_pred, u_true, v_true, dot_gz, dot_uv)

        # 保存诊断指标
        self.save_diagnostics(final_losses, g_dot_gradZ0=dot_gz, cos_sim=dot_uv, extra_metrics=extra_metrics)

        # 输出综合诊断图像（预测 vs 真实 + 残差直方图）
        self.plot_uv_diagnostics(u_pred, u_true, v_pred, v_true)

        print("所有输出已保存完毕")