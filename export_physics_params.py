"""
Example Usage (PowerShell):
    python export_physics_params.py `
      --input-csv "data/processed_data_mean_test.csv" `
      --train-csv "data/processed_data_mean_train.csv" `
      --model "outputs/exp_1/model_final.pt" `
      --output-dir "outputs"

Exports learned physical quantities using a trained model checkpoint (no retraining required).
- Requires an input CSV and trained model weights.
- Optionally fits scaler on the training set CSV for normalization consistency.
- Saves: 
  - physics_fields.csv (per-point physical fields)
  - physics_scalars.json (global scalar parameters)
  - full_prediction.csv (if not already present)
"""

import argparse
import os
import torch
import pandas as pd

from config import ModelConfig
from scaler_manager import ScalerManager
from physics_model import EnhancedPhysicsInformedThermocline
from data_loader import load_csv_data_from_df
from compute_approximate_velocity import compute_velocity
from save_utils import PredictionSaver

def main():
    parser = argparse.ArgumentParser(
        description="Export learned physical quantities from a trained model (no retraining required)."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV data file (usually test set).")
    parser.add_argument("--model", required=True, help="Path to trained model weights (e.g., model_final.pt).")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--train-csv", default=None, help="(Optional) Training set CSV for scaler fitting (should match training normalization).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fit or reuse scaler for consistent normalization
    if args.train_csv is not None:
        train_df = pd.read_csv(args.train_csv)
        _, _, _, _, _, _, scaler_mgr_train, _ = load_csv_data_from_df(train_df, device=device, fit_scaler=True)
        input_df = pd.read_csv(args.input_csv)
        t, x, y, z, u_true, v_true, scaler_mgr, original_df = load_csv_data_from_df(
            input_df, device=device, scaler_mgr=scaler_mgr_train, fit_scaler=False
        )
    else:
        input_df = pd.read_csv(args.input_csv)
        t, x, y, z, u_true, v_true, scaler_mgr, original_df = load_csv_data_from_df(
            input_df, device=device, fit_scaler=True
        )

    # Build model (must use identical normalization as in training phase)
    config = ModelConfig(
        Ro=0.024,  # Initial value, will be overwritten by trained checkpoint
        omega_0=30.0,
        use_scaler=True,
        scaler_mgr=scaler_mgr,
        depth_scaler=scaler_mgr.depth_scaler,
        velocity_scale=1.0,
        grad_clip=10.0
    )

    # Load trained weights
    model = EnhancedPhysicsInformedThermocline(config).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Run model prediction and extract physical quantities
    x.requires_grad_(True)
    y.requires_grad_(True)
    (
        u_pred, v_pred, u_bar, v_bar, theta, eta, Z0,
        P, g1, g2, h1, h2, time_phase_C2, time_phase_C3
    ) = compute_velocity(model, x, y, z, t)

    # Save predictions and physical fields
    saver = PredictionSaver(model, scaler_mgr, args.output_dir)

    # Optional: save full prediction results with matching columns
    saver.save_full_prediction(
        original_df=original_df,
        u_pred=u_pred, v_pred=v_pred,
        u_true=u_true, v_true=v_true,
        Z0=Z0, x=x, y=y, z=z, t=t
    )

    # Save per-point physical fields
    saver.save_physics_fields(
        x=x, y=y, z=z, t=t,
        Z0=Z0, theta=theta, eta=eta,
        P=P, g1=g1, g2=g2, h1=h1, h2=h2,
        time_phase_C2=time_phase_C2, time_phase_C3=time_phase_C3
    )

    # Save global scalar parameters
    saver.save_physics_scalars()

    print("[OK] Exported: physics_fields.csv and physics_scalars.json.")

if __name__ == "__main__":
    main()