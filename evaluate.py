#!/usr/bin/env python3
# evaluate.py
# Description: Generate test set predictions and compute MAE statistics using a trained model (by variable and by day).
# Example usage:
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

# ----------------- Utilities: angle difference, date parsing, metric calculation -----------------
def _angle_diff_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b)
    d = np.mod(d, 2 * np.pi)
    d = np.minimum(d, 2 * np.pi - d)
    return d

def _ensure_date_series_from_df(df: pd.DataFrame, date_col_hint: str = "date"):
    """
    Return a date series (YYYY-MM-DD) aligned with df, or None if unavailable.
    """
    if date_col_hint in df.columns:
        s = pd.to_datetime(df[date_col_hint], errors="coerce")
        if s.isna().all():
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
    Compute and save summary MAE by variable and (if available) by day.
    Expects df_cmp to contain columns: u_true, u_pred, v_true, v_pred.
    """
    required = {"u_true", "u_pred", "v_true", "v_pred"}
    if not required.issubset(set(df_cmp.columns)):
        raise ValueError(f"comparison CSV must contain columns: {required}")

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
    theta_abs = _angle_diff_rad(ang_pred, ang_true)
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
        try:
            per_day["date_parsed"] = pd.to_datetime(per_day["date"])
            per_day = per_day.sort_values("date_parsed").drop(columns=["date_parsed"])
        except Exception:
            per_day = per_day.sort_values("date")
        per_day_path = os.path.join(outdir, "eval_per_day.csv")
        per_day.to_csv(per_day_path, index=False, encoding="utf-8")
        print(f"[INFO] Daily metrics saved: {per_day_path}")
    else:
        print("[INFO] No date info found, eval_per_day.csv was not generated.")

    print(f"[INFO] Per-variable metrics saved: {per_var_path}")
    return per_var_df

# ----------------- Main pipeline -----------------
def main():
    p = argparse.ArgumentParser(description="Generate test set predictions and MAE using a trained model (no retraining).")
    p.add_argument("--model", required=True, help="Path to trained model file (model_final.pt)")
    p.add_argument("--train-csv", required=True, help="Training set CSV (for fitting scaler, must match training)")
    p.add_argument("--test-csv", required=True, help="Test set CSV (for prediction)")
    p.add_argument("--outdir", default=None, help="Output directory for comparison and results (default: model directory)")
    p.add_argument("--device", default="cpu", help="Device, e.g. cuda:0 or cpu")
    args = p.parse_args()

    model_path = args.model
    train_csv = args.train_csv
    test_csv = args.test_csv
    outdir = args.outdir or os.path.dirname(os.path.abspath(model_path))
    os.makedirs(outdir, exist_ok=True)
    device = torch.device(args.device)

    # 1) Fit the scaler on training data, to ensure normalization matches training phase
    print("[INFO] Fitting scaler on training CSV...")
    train_df = pd.read_csv(train_csv)
    _, _, _, _, _, _, scaler_mgr, _ = load_csv_data_from_df(train_df, device=device, fit_scaler=True)

    # 2) Normalize test set using the fitted scaler
    print("[INFO] Preparing test set with training scaler...")
    test_df = pd.read_csv(test_csv)
    t_test, x_test, y_test, z_test, u_true_test, v_true_test, _, original_test_df = load_csv_data_from_df(
        test_df, device=device, scaler_mgr=scaler_mgr, fit_scaler=False
    )

    # 3) Construct model and load weights
    print("[INFO] Constructing model and loading weights...")
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
    sd = torch.load(model_path, map_location=device)
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    try:
        model.load_state_dict(sd)
    except Exception:
        if "state_dict" in sd:
            state = sd["state_dict"]
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state)
        else:
            raise
    model.to(device)
    model.eval()

    # 4) Generate predictions on test set
    print("[INFO] Generating predictions on test set...")
    with torch.no_grad():
        (u_pred_test, v_pred_test, u_bar, v_bar, theta, eta, Z0,
         P, g1, g2, h1, h2, time_phase_C2, time_phase_C3) = compute_velocity(
            model, x_test, y_test, z_test, t_test
        )

    u_pred_np = u_pred_test.detach().cpu().numpy().flatten()
    v_pred_np = v_pred_test.detach().cpu().numpy().flatten()
    u_true_np = u_true_test.detach().cpu().numpy().flatten()
    v_true_np = v_true_test.detach().cpu().numpy().flatten()

    # Assemble the comparison DataFrame (with date if present)
    comp_df = pd.DataFrame({
        "u_true": u_true_np,
        "u_pred": u_pred_np,
        "v_true": v_true_np,
        "v_pred": v_pred_np
    })

    # Try to preserve date/time columns, if present
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

    cmp_path = os.path.join(outdir, "velocity_comparison_data.csv")
    comp_df.to_csv(cmp_path, index=False, encoding="utf-8")
    print(f"[INFO] Prediction comparison saved: {cmp_path}")

    # 5) Compute and save MAE metrics
    print("[INFO] Computing MAE metrics and writing files...")
    date_series_for_metrics = comp_df["date"] if "date" in comp_df.columns else None
    compute_metrics_and_save(comp_df, date_series_for_metrics, outdir)

    print("[INFO] Evaluation complete.")

if __name__ == "__main__":
    main()