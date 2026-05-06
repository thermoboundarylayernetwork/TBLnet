#!/usr/bin/env python3
# evaluate.py
# 说明：使用已训练好的 model_final.pt 在测试集上生成预测并计算 MAE（按变量与按天）。
# 用法示例：
# python evaluate.py --model outputs/exp_1/model_final.pt --train-csv data/processed_data_mean_train.csv --test-csv data/processed_data_mean_test.csv --outdir outputs/exp_1 --device cuda:0

import argparse
import os
import pandas as pd
import numpy as np
import torch

from data_loader import load_csv_data_from_df
from config import ModelConfig
from physics_model import EnhancedPhysicsInformedThermocline
from compute_approximate_velocity import compute_velocity

# ----------------- 辅助函数：角度差、日期列处理、指标计算 -----------------
def _angle_diff_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b)
    d = np.mod(d, 2 * np.pi)
    d = np.minimum(d, 2 * np.pi - d)
    return d

def _ensure_date_series_from_df(df: pd.DataFrame, date_col_hint: str = "date"):
    """
    返回与 df 行对齐的日期字符串 Series (YYYY-MM-DD) 或 None（如果找不到）。
    """
    if date_col_hint in df.columns:
        s = pd.to_datetime(df[date_col_hint], errors="coerce")
        if s.isna().all():
            # 如果解析失败，退回字符串形式（仍可用）
            return df[date_col_hint].astype(str)
        return s.dt.date.astype(str)
    for c in ("time", "datetime", "timestamp"):
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if not s.isna().all():
                return s.dt.date.astype(str)
    return None

def compute_metrics_and_save(df_cmp: pd.DataFrame, date_series: pd.Series | None, outdir: str):
    """
    df_cmp 必须包含列 u_true,u_pred,v_true,v_pred
    生成并保存 eval_per_var.csv 和（如有日期）eval_per_day.csv
    """
    required = {"u_true", "u_pred", "v_true", "v_pred"}
    if not required.issubset(set(df_cmp.columns)):
        raise ValueError(f"comparison CSV 必须包含列: {required}")

    u_true = df_cmp["u_true"].to_numpy(dtype=float)
    v_true = df_cmp["v_true"].to_numpy(dtype=float)
    u_pred = df_cmp["u_pred"].to_numpy(dtype=float)
    v_pred = df_cmp["v_pred"].to_numpy(dtype=float)

    u_abs = np.abs(u_pred - u_true)
    v_abs = np.abs(v_pred - v_true)
    speed_true = np.sqrt(u_true**2 + v_true**2)
    speed_pred = np.sqrt(u_pred**2 + v_pred**2)
    so_abs = np.abs(speed_pred - speed_true)
    ang_true = np.arctan2(v_true, u_true)
    ang_pred = np.arctan2(v_pred, u_pred)
    theta_abs = _angle_diff_rad(ang_pred, ang_true)  # 弧度
    overall_abs = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)

    per_var = [
        ("so", float(np.mean(so_abs)), int(len(so_abs))),
        ("thetao", float(np.mean(theta_abs)), int(len(theta_abs))),
        ("uo", float(np.mean(u_abs)), int(len(u_abs))),
        ("vo", float(np.mean(v_abs)), int(len(v_abs))),
    ]
    per_var_df = pd.DataFrame(per_var, columns=["variable", "mae", "count"])
    per_var_path = os.path.join(outdir, "eval_per_var.csv")
    per_var_df.to_csv(per_var_path, index=False, encoding="utf-8")

    if date_series is not None:
        df_metrics = pd.DataFrame({
            "date": date_series.astype(str),
            "so_abs": so_abs,
            "theta_abs": theta_abs,
            "uo_abs": u_abs,
            "vo_abs": v_abs,
            "overall_abs": overall_abs
        })
        per_day = df_metrics.groupby("date").mean().reset_index().rename(columns={
            "so_abs": "mae_so",
            "theta_abs": "mae_thetao",
            "uo_abs": "mae_uo",
            "vo_abs": "mae_vo",
            "overall_abs": "mae_overall"
        })
        # 尝试按日期排序
        try:
            per_day["date_parsed"] = pd.to_datetime(per_day["date"])
            per_day = per_day.sort_values("date_parsed").drop(columns=["date_parsed"])
        except Exception:
            per_day = per_day.sort_values("date")
        per_day_path = os.path.join(outdir, "eval_per_day.csv")
        per_day.to_csv(per_day_path, index=False, encoding="utf-8")
        print(f"已保存按天结果: {per_day_path}")
    else:
        print("未找到日期信息，未生成 eval_per_day.csv")

    print(f"已保存按变量结果: {per_var_path}")
    return per_var_df

# ----------------- 主流程 -----------------
def main():
    p = argparse.ArgumentParser(description="用已训练模型在测试集上生成预测并计算 MAE（不重新训练）")
    p.add_argument("--model", required=True, help="已训练好的模型文件（model_final.pt）路径")
    p.add_argument("--train-csv", required=True, help="训练集 CSV（用于 fit Scaler，与训练时一致）")
    p.add_argument("--test-csv", required=True, help="测试集 CSV（用于生成预测和保存 comparison）")
    p.add_argument("--outdir", default=None, help="输出目录（保存 comparison 与 eval 文件），默认为模型所在目录")
    p.add_argument("--device", default="cpu", help="设备，如 cuda:0 或 cpu")
    args = p.parse_args()

    model_path = args.model
    train_csv = args.train_csv
    test_csv = args.test_csv
    outdir = args.outdir or os.path.dirname(os.path.abspath(model_path))
    os.makedirs(outdir, exist_ok=True)
    device = torch.device(args.device)

    # 1) 读取训练 CSV 并 fit Scaler（使用 data_loader 的函数以保证一致）
    print("读取训练文件并拟合 Scaler...")
    train_df = pd.read_csv(train_csv)
    # load_csv_data_from_df(..., fit_scaler=True) 会 fit 并返回 scaler_mgr
    _, _, _, _, _, _, scaler_mgr, _ = load_csv_data_from_df(train_df, device=device, fit_scaler=True)

    # 2) 读取测试 CSV 并使用训练的 scaler 做 transform
    print("读取测试文件并使用训练时的 Scaler 做变换...")
    test_df = pd.read_csv(test_csv)
    t_test, x_test, y_test, z_test, u_true_test, v_true_test, _, original_test_df = load_csv_data_from_df(
        test_df, device=device, scaler_mgr=scaler_mgr, fit_scaler=False
    )

    # 3) 构建模型并加载权重
    print("构建模型并加载权重...")
    config = ModelConfig(
        Ro=0.024,
        omega_0=30.0,
        use_scaler=True,
        scaler_mgr=scaler_mgr,
        depth_scaler=scaler_mgr.depth_scaler,
        velocity_scale=1.0,
        grad_clip=10.0
    )
    model = EnhancedPhysicsInformedThermocline(config)
    # 加载权重（支持 DataParallel 保存的 'module.' 前缀）
    sd = torch.load(model_path, map_location=device)
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    try:
        model.load_state_dict(sd)
    except Exception:
        # 有些保存会把 dict 放在 key 'state_dict' 或其它键下，尝试适配
        if "state_dict" in sd:
            state = sd["state_dict"]
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state)
        else:
            raise
    model.to(device)
    model.eval()

    # 4) 用模型生成预测
    print("生成测试集预测（compute_velocity）...")
    with torch.no_grad():
        (u_pred_test, v_pred_test, u_bar, v_bar, theta, eta, Z0,
         P, g1, g2, h1, h2, time_phase_C2, time_phase_C3) = compute_velocity(
            model, x_test, y_test, z_test, t_test
        )

    # 转为 numpy 存储（拉到 cpu）
    u_pred_np = u_pred_test.detach().cpu().numpy().flatten()
    v_pred_np = v_pred_test.detach().cpu().numpy().flatten()
    u_true_np = u_true_test.detach().cpu().numpy().flatten()
    v_true_np = v_true_test.detach().cpu().numpy().flatten()

    # 组装 comparison DataFrame（包含可能的 date 列）
    comp_df = pd.DataFrame({
        "u_true": u_true_np,
        "u_pred": u_pred_np,
        "v_true": v_true_np,
        "v_pred": v_pred_np
    })

    # 尝试复制 test 原始的日期列（若存在）
    date_series = None
    for cand in ("date", "time", "datetime", "timestamp"):
        if cand in original_test_df.columns:
            try:
                date_series = pd.to_datetime(original_test_df[cand], errors="coerce")
                comp_df["date"] = date_series.astype(str)
                break
            except Exception:
                comp_df["date"] = original_test_df[cand].astype(str)
                break

    # 保存 comparison CSV
    cmp_path = os.path.join(outdir, "velocity_comparison_data.csv")
    comp_df.to_csv(cmp_path, index=False, encoding="utf-8")
    print(f"已保存预测结果对比文件: {cmp_path}")

    # 5) 计算并保存 eval_per_var.csv 和 eval_per_day.csv（如有日期）
    print("计算 MAE 指标并保存...")
    date_series_for_metrics = comp_df["date"] if "date" in comp_df.columns else None
    compute_metrics_and_save(comp_df, date_series_for_metrics, outdir)

    print("全部完成。")

if __name__ == "__main__":
    main()