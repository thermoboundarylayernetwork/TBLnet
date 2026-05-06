'''
    powershell调用命令：python export_physics_params.py `
  --input-csv "E:\Python_where\Python\PythonProject\mar_PINN-2\data\processed_data_mean_test.csv" `
  --train-csv "E:\Python_where\Python\PythonProject\mar_PINN-2\data\processed_data_mean_train.csv" `
  --model "E:\Python_where\Python\PythonProject\mar_PINN-2\outputs\exp_1\model_final.pt" `
  --output-dir "E:\Python_where\Python\PythonProject\mar_PINN-2\outputs"
'''


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
    parser = argparse.ArgumentParser(description="导出 PINN-2 学习到的物理参数（无需重新训练）")
    parser.add_argument("--input-csv", required=True, help="用于导出的数据 CSV（通常为测试集）")
    parser.add_argument("--model", required=True, help="已训练好的模型权重路径（如 model_final.pt）")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--train-csv", default=None, help="可选，训练集 CSV（用于拟合 scaler，与训练时一致）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 拟合/复用 scaler，与 evaluate.py 逻辑保持一致
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

    # 模型配置（与训练保持一致，use_scaler=True）
    config = ModelConfig(
        Ro=0.024,  # 初始化会被已训练权重覆盖
        omega_0=30.0,
        use_scaler=True,
        scaler_mgr=scaler_mgr,
        depth_scaler=scaler_mgr.depth_scaler,
        velocity_scale=1.0,
        grad_clip=10.0
    )

    # 加载模型
    model = EnhancedPhysicsInformedThermocline(config).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 预测并提取物理量
    x.requires_grad_(True)
    y.requires_grad_(True)
    (
        u_pred, v_pred, u_bar, v_bar, theta, eta, Z0,
        P, g1, g2, h1, h2, time_phase_C2, time_phase_C3
    ) = compute_velocity(model, x, y, z, t)

    # 保存：完整预测（含 u/v/Z0）+ 物理场 + 标量参数
    saver = PredictionSaver(model, scaler_mgr, args.output_dir)

    # 可选：保存完整预测（含原始字段对齐）
    saver.save_full_prediction(
        original_df=original_df,
        u_pred=u_pred, v_pred=v_pred,
        u_true=u_true, v_true=v_true,
        Z0=Z0, x=x, y=y, z=z, t=t
    )

    # 保存物理场逐点输出
    saver.save_physics_fields(
        x=x, y=y, z=z, t=t,
        Z0=Z0, theta=theta, eta=eta,
        P=P, g1=g1, g2=g2, h1=h1, h2=h2,
        time_phase_C2=time_phase_C2, time_phase_C3=time_phase_C3
    )

    # 保存全局标量参数
    saver.save_physics_scalars()

    print("✅ 导出完成：physics_fields.csv 与 physics_scalars.json 已生成。")

if __name__ == "__main__":
    main()