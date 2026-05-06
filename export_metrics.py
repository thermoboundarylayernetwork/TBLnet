#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage
-----
1) Basic run:
  python export_metrics.py --pred outputs/exp_9/full_prediction.csv --outdir outputs/exp_9

2) With test (optional; only checks the file is readable, not used for metrics):
  python export_metrics.py --pred outputs/exp_9/full_prediction.csv --test data/processed_data_mean_test.csv --outdir outputs/exp_9

Outputs
-------
- <outdir>/exported_metrics.csv  (if locked, writes exported_metrics_<timestamp>.csv)
- <outdir>/exported_metrics.json (if locked, writes exported_metrics_<timestamp>.json)

Metrics computed
----------------
A) (ĝ · ∇Ẑ0) from physics_fields.csv
   - Uses columns: g1, g2, Z0_x, Z0_y
   - Normalize g and ∇Z0 first, then dot:
       g_hat = g / (||g|| + eps)
       grad_hat = grad / (||grad|| + eps)
       dot = g_hat · grad_hat
   - Exports mean/std/min/max.

B) mean(arccos(u_p · u_t)) WITHOUT normalization (as requested)
   dot_i = u_pred_i*u_true_i + v_pred_i*v_true_i
   theta_i = arccos(clip(dot_i, -1, 1))
   report mean/std of {theta_i} in rad and deg
   (raw_dot_clip_frac removed)

C) u_pred-u_true and v_pred-v_true:
   - MAE
   - RMSE (replaces std)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ------------------------- stats helpers -------------------------

def _np(a) -> np.ndarray:
    return a if isinstance(a, np.ndarray) else np.asarray(a)


def mae(x: np.ndarray) -> float:
    x = _np(x).astype(float)
    return float(np.mean(np.abs(x)))


def rmse(x: np.ndarray) -> float:
    x = _np(x).astype(float)
    return float(np.sqrt(np.mean(x * x)))


# ------------------------- output -------------------------

@dataclass
class MetricRow:
    metric: str
    stat: str
    value: float


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_outputs(outdir: Path, rows: List[MetricRow]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame([asdict(r) for r in rows])

    base_csv = outdir / "exported_metrics.csv"
    base_json = outdir / "exported_metrics.json"

    # CSV (fallback if locked)
    csv_path = base_csv
    try:
        df_out.to_csv(csv_path, index=False, encoding="utf-8")
    except PermissionError:
        csv_path = outdir / f"exported_metrics_{_timestamp()}.csv"
        df_out.to_csv(csv_path, index=False, encoding="utf-8")

    # JSON (fallback if locked)
    payload = {}
    for r in rows:
        payload.setdefault(r.metric, {})[r.stat] = r.value

    json_path = base_json
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except PermissionError:
        json_path = outdir / f"exported_metrics_{_timestamp()}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {json_path}")


# ------------------------- main -------------------------

def main():
    p = argparse.ArgumentParser(description="Export metrics from full_prediction.csv (exp_9 by default)")
    p.add_argument("--pred", default="outputs/exp_9/full_prediction.csv", help="Prediction CSV path")
    p.add_argument("--test", default=None, help="Optional test CSV path (only checked readable)")
    p.add_argument("--outdir", default="outputs/exp_9", help="Output directory")

    # for normalized dot products
    p.add_argument("--eps", type=float, default=1e-12, help="Epsilon to avoid divide-by-zero in normalizations")

    args = p.parse_args()

    pred_df = pd.read_csv(Path(args.pred))

    # required columns for residuals + angle metric
    required = ["u_pred", "v_pred", "u_true", "v_true"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Prediction file missing required columns: {missing}")

    rows: List[MetricRow] = []

    # (C) residuals: u_pred - u_true, v_pred - v_true
    du = pred_df["u_pred"].to_numpy() - pred_df["u_true"].to_numpy()
    dv = pred_df["v_pred"].to_numpy() - pred_df["v_true"].to_numpy()
    rows += [
        MetricRow("u_pred_minus_u_true", "mae", mae(du)),
        MetricRow("u_pred_minus_u_true", "rmse", rmse(du)),
        MetricRow("v_pred_minus_v_true", "mae", mae(dv)),
        MetricRow("v_pred_minus_v_true", "rmse", rmse(dv)),
    ]

    # (B) mean(arccos(u_p · u_t)) WITHOUT normalization
    dot = (
        pred_df["u_pred"].to_numpy() * pred_df["u_true"].to_numpy()
        + pred_df["v_pred"].to_numpy() * pred_df["v_true"].to_numpy()
    ).astype(float)

    dot_clip = np.clip(dot, -1.0, 1.0)

    # keep basic diagnostics but remove raw_dot_clip_frac as requested
    rows += [
        MetricRow("raw_dot_mean", "value", float(np.mean(dot))),
        MetricRow("raw_dot_clip_mean", "value", float(np.mean(dot_clip))),
    ]

    # first arccos per-sample, then mean/std
    ang_each = np.arccos(dot_clip)  # radians
    ang_each_deg = ang_each * 180.0 / np.pi

    rows += [
        MetricRow("mean_of_arccos_u_pred_dot_u_true_raw_rad", "mean", float(np.mean(ang_each))),
        MetricRow("mean_of_arccos_u_pred_dot_u_true_raw_rad", "std", float(np.std(ang_each))),
        MetricRow("mean_of_arccos_u_pred_dot_u_true_raw_deg", "mean", float(np.mean(ang_each_deg))),
        MetricRow("mean_of_arccos_u_pred_dot_u_true_raw_deg", "std", float(np.std(ang_each_deg))),
    ]

    # (A) normalize then dot: g_hat · gradZ0_hat  (from physics_fields.csv)
    physics_fields_path = Path(args.outdir) / "physics_fields.csv"
    if physics_fields_path.exists():
        df_phys = pd.read_csv(physics_fields_path)

        need_cols = ["g1", "g2", "Z0_x", "Z0_y"]
        miss_cols = [c for c in need_cols if c not in df_phys.columns]
        if miss_cols:
            print(f"[WARN] {physics_fields_path} missing columns {miss_cols}, skip normalized g·∇Z0.")
        else:
            g1 = df_phys["g1"].to_numpy(dtype=float)
            g2 = df_phys["g2"].to_numpy(dtype=float)
            zx = df_phys["Z0_x"].to_numpy(dtype=float)
            zy = df_phys["Z0_y"].to_numpy(dtype=float)

            g_norm = np.sqrt(g1 * g1 + g2 * g2)
            grad_norm = np.sqrt(zx * zx + zy * zy)

            eps = float(args.eps)
            g1h = g1 / (g_norm + eps)
            g2h = g2 / (g_norm + eps)
            zxh = zx / (grad_norm + eps)
            zyh = zy / (grad_norm + eps)

            gd_normed = g1h * zxh + g2h * zyh  # cosine-like similarity in [-1,1]

            rows += [
                MetricRow("g_hat_dot_gradZ0_hat", "mean", float(np.mean(gd_normed))),
                MetricRow("g_hat_dot_gradZ0_hat", "std", float(np.std(gd_normed))),
                MetricRow("g_hat_dot_gradZ0_hat", "min", float(np.min(gd_normed))),
                MetricRow("g_hat_dot_gradZ0_hat", "max", float(np.max(gd_normed))),
            ]
    else:
        print(f"[WARN] {physics_fields_path} not found, skip normalized g·∇Z0.")

    # optional test read (not used)
    if args.test:
        _ = pd.read_csv(Path(args.test), nrows=1)

    write_outputs(Path(args.outdir), rows)


if __name__ == "__main__":
    main()