"""
Usage
-----
1) Basic run:
  python export_metrics.py --pred outputs/exp_9/full_prediction.csv --outdir outputs/exp_9

2) With test file (optional, only checks readability; metrics are not computed on it):
  python export_metrics.py --pred outputs/exp_9/full_prediction.csv --test data/processed_data_mean_test.csv --outdir outputs/exp_9

Outputs
-------
- <outdir>/exported_metrics.csv  (if locked, writes exported_metrics_<timestamp>.csv)
- <outdir>/exported_metrics.json (if locked, writes exported_metrics_<timestamp>.json)

Metrics computed
----------------
A) (ĝ · ∇Ẑ0) from physics_fields.csv
   - Uses columns: g1, g2, Z0_x, Z0_y
   - Normalizes g and ∇Z0, then computes dot:
       g_hat = g / (||g|| + eps)
       grad_hat = grad / (||grad|| + eps)
       dot = g_hat · grad_hat
   - Exports mean/std/min/max.

B) mean(arccos(u_p · u_t)) WITHOUT normalization
   dot_i = u_pred_i*u_true_i + v_pred_i*v_true_i
   theta_i = arccos(clip(dot_i, -1, 1))
   Reports mean/std of {theta_i} in radians and degrees.

C) u_pred-u_true and v_pred-v_true:
   - MAE
   - RMSE
"""

# [ ... rest of the code unchanged ... ]