"usage: python train.py --train data\processed_data_mean_train.csv --test data\processed_data_mean_test.csv --output outputs\exp_8 --log-loss --device cuda:0"

import argparse
import os
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm.auto import tqdm

from losses import direction_loss, value_loss, activation_loss
from config import ModelConfig
from data_loader import split_dataset_random, load_csv_data_from_df
from physics_model import EnhancedPhysicsInformedThermocline
from compute_approximate_velocity import compute_velocity
from compute_physics_residual import compute_residuals
from physics_residual import geometric_constraint
from utils import get_device, prepare_inputs, LossManager
from save_utils import PredictionSaver


def parse_args_for_train():
    p = argparse.ArgumentParser()
    # 支持显式传入训练/测试文件；为兼容之前的单 file 参数，也保留 --input（如只传 input 则仍会做随机划分）
    p.add_argument("--input", "-i", default=None,
                   help="若只传入一个文件，将对其随机划分为训练/测试（向后兼容）。")
    p.add_argument("--train", default=None, help="训练集 CSV 文件（优先于 --input）")
    p.add_argument("--test", default=None, help="测试集 CSV 文件（与 --train 一起使用）")
    p.add_argument("--output", "-o", default="outputs/run1",
                   help="训练输出目录（默认: outputs/run1）")
    p.add_argument("--device", default=None, help="可选：cuda:0 或 cpu")

    # 新增：loss 曲线记录
    p.add_argument("--log-loss", action="store_true",
                   help="记录训练过程各分量损失到 CSV（用于画收敛曲线）。")
    p.add_argument("--log-loss-every", type=int, default=1,
                   help="每隔多少个 epoch 记录一次损失（默认: 1，每轮都记；设大一点可减小文件大小）。")

    return p.parse_args()


def train_prediction_model(train_path=None, test_path=None, input_path=None, output_dir="outputs/run1", device=None,
                           log_loss: bool = False, log_loss_every: int = 1):
    os.makedirs(output_dir, exist_ok=True)

    if device is not None:
        device = torch.device(device)
    else:
        device = get_device()

    # -------- 加载训练/测试数据的逻辑 --------
    if train_path is not None and test_path is not None:
        # 直接从文件读取（你现在的场景）
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # 训练集：fit scaler 并返回 scaler_mgr
        t, x, y, z, u_true, v_true, scaler_mgr, original_df = load_csv_data_from_df(train_df, device=device, fit_scaler=True)

        # 测试集：复用训练时的 scaler_mgr（fit_scaler=False）
        t_test, x_test, y_test, z_test, u_true_test, v_true_test, _, original_test_df = load_csv_data_from_df(
            test_df, device=device, scaler_mgr=scaler_mgr, fit_scaler=False
        )

    elif input_path is not None:
        # 向后兼容：只传入单个文件，使用随机划分
        train_df, test_df = split_dataset_random(input_path)
        t, x, y, z, u_true, v_true, scaler_mgr, original_df = load_csv_data_from_df(train_df, device=device, fit_scaler=True)
        t_test, x_test, y_test, z_test, u_true_test, v_true_test, _, original_test_df = load_csv_data_from_df(
            test_df, device=device, scaler_mgr=scaler_mgr, fit_scaler=False
        )
    else:
        raise ValueError("请传入 --train 与 --test（首选），或传入单个 --input 以随机划分。")

    # ============ 以下为原训练逻辑（保持不变，仅使用上面准备好的 t,x,y,z,u_true,v_true 变量） ============
    config = ModelConfig(
        Ro=0.024,
        omega_0=30.0,
        use_scaler=True,
        scaler_mgr=scaler_mgr,
        depth_scaler=scaler_mgr.depth_scaler,
        velocity_scale=1.0,
        grad_clip=10.0
    )
    model = EnhancedPhysicsInformedThermocline(config).to(device)

    # 学习率
    base_lr = 1e-4
    ro_base_lr = 5e-4

    ro_params = [p for n, p in model.named_parameters() if n == 'Ro']
    other_params = [p for n, p in model.named_parameters() if n != 'Ro']

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': base_lr},
        {'params': ro_params, 'lr': ro_base_lr}
    ])

    loss_manager = LossManager({
        'data': 5.0, 'dir': 5, 'phys': 3.0,
        'cont': 1.0, 'geo': 0.2, 'act': 5
    })

    def get_dynamic_weights(epoch):
        if epoch < 3000:
            ro_w = 0.5 * (1 - 0.5 * epoch / 3000)
        elif epoch < 6000:
            ro_w = 0.25 * (1 - 0.6 * (epoch - 3000) / 3000)
        else:
            ro_w = 0.1
        return ro_w, 0.02

    def update_ro_lr(epoch):
        initial_ro_weight = 0.5
        ro_weight, _ = get_dynamic_weights(epoch)
        lr_scale = max(0.1, ro_weight / initial_ro_weight)
        for pg in optimizer.param_groups:
            if len(pg['params']) == 1 and pg['params'][0] is getattr(model, 'Ro', None):
                pg['lr'] = ro_base_lr * lr_scale
                break
        return lr_scale

    def compute_ro_regularization(model, ro_weight, b_weight):
        ro_min, ro_max = 0.02, 0.1
        ro_reg = ro_weight * (
            F.softplus(50 * (ro_min - model.Ro)) +
            F.softplus(50 * (model.Ro - ro_max))
        )
        b0_reg = b_weight * F.softplus(20 * (model.B0_scalar - 0.05))
        b1_reg = b_weight * F.softplus(20 * (model.B1_scalar - 0.05))
        total_reg = ro_reg + b0_reg + b1_reg
        return total_reg, ro_reg, b0_reg, b1_reg

    # loss 日志：用 list 收集，训练结束统一写 CSV
    loss_rows = []
    loss_csv_path = os.path.join(output_dir, "loss_history.csv")

    ro_history = []
    num_epochs = 6000
    final_train_loss = None

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in pbar:
        ro_lr_scale = update_ro_lr(epoch)
        ro_weight, b_weight = get_dynamic_weights(epoch)

        if epoch >= 3000 and epoch % 500 == 0:
            step = (epoch - 3000) // 500
            loss_manager.set_weights({
                'phys': min(5.0, 3.0 + 0.2 * step),
                'cont': min(2.0, 1.0 + 0.1 * step),
                'geo': min(0.5, 0.2 + 0.05 * step),
                'dir': min(8.0, 5.0 + 0.2 * step)
            })

        optimizer.zero_grad()
        x, y, z, t = prepare_inputs(x, y, z, t)

        (u_pred, v_pred, u_bar, v_bar, theta, eta, Z0,
         P, g1, g2, h1, h2, time_phase_C2, time_phase_C3) = compute_velocity(model, x, y, z, t)

        losses = {
            'data': value_loss(u_pred, v_pred, u_true, v_true),
            'dir': direction_loss(u_pred, v_pred, u_true, v_true),
            'act': activation_loss(theta, eta),
            'geo': geometric_constraint(g1, g2, Z0, x, y),
        }

        res_u, res_v, res_cont = compute_residuals(model, x, y, z, t)
        losses['phys'] = torch.mean(res_u ** 2 + res_v ** 2)
        losses['cont'] = torch.mean(res_cont ** 2)

        grad_z0 = [torch.autograd.grad(Z0, v, grad_outputs=torch.ones_like(Z0),
                                       create_graph=True)[0] for v in [x, y]]
        z0_reg = (0.5 * torch.mean(Z0) ** 2 +
                  0.2 * (torch.var(Z0) - 0.1) ** 2 +
                  0.1 * torch.mean(sum(g ** 2 for g in grad_z0)))

        reg_total, ro_reg, b0_reg, b1_reg = compute_ro_regularization(model, ro_weight, b_weight)

        total_loss = loss_manager.compute_total_loss(losses, epoch=epoch)
        total_loss = total_loss + reg_total + z0_reg

        ro_change_penalty = torch.tensor(0.0, device=device)
        if epoch > 7000 and ro_history:
            ro_change_penalty = 0.1 * (model.Ro - ro_history[-1]) ** 2
            total_loss = total_loss + ro_change_penalty

        total_loss.backward()

        if hasattr(model, 'Ro') and model.Ro.grad is not None and torch.abs(model.Ro.grad) > 1.0:
            model.Ro.grad.data.clamp_(-0.5, 0.5)

        optimizer.step()
        with torch.no_grad():
            if hasattr(model, 'Ro'):
                model.Ro.clamp_(1e-2, 0.5)

        if torch.isnan(total_loss):
            pbar.write(f"[Epoch {epoch}] NaN detected, stopping.")
            final_train_loss = float('nan')
            break

        ro_history.append(model.Ro.item() if hasattr(model, 'Ro') else float('nan'))
        final_train_loss = total_loss.item()

        # 记录损失（用于画收敛曲线）
        if log_loss and (epoch % max(1, log_loss_every) == 0):
            row = {
                'epoch': int(epoch),
                'total': float(total_loss.detach().cpu().item()),
                'data': float(losses['data'].detach().cpu().item()),
                'dir': float(losses['dir'].detach().cpu().item()),
                'phys': float(losses['phys'].detach().cpu().item()),
                'cont': float(losses['cont'].detach().cpu().item()),
                'geo': float(losses['geo'].detach().cpu().item()),
                'act': float(losses['act'].detach().cpu().item()),
                'z0_reg': float(z0_reg.detach().cpu().item()),
                'ro_reg': float(ro_reg.detach().cpu().item()),
                'b0_reg': float(b0_reg.detach().cpu().item()),
                'b1_reg': float(b1_reg.detach().cpu().item()),
                'ro_change_penalty': float(ro_change_penalty.detach().cpu().item()),
                'Ro': float(model.Ro.detach().cpu().item()) if hasattr(model, 'Ro') else float('nan'),
                'lr_other': float(next(
                    (pg['lr'] for pg in optimizer.param_groups
                     if not (len(pg['params']) == 1 and pg['params'][0] is getattr(model, 'Ro', None))),
                    base_lr
                )),
                'lr_Ro': float(next(
                    (pg['lr'] for pg in optimizer.param_groups
                     if len(pg['params']) == 1 and pg['params'][0] is getattr(model, 'Ro', None)),
                    ro_base_lr
                )),
            }
            loss_rows.append(row)

        try:
            current_lr_other = next(
                (pg['lr'] for pg in optimizer.param_groups
                 if not (len(pg['params']) == 1 and pg['params'][0] is getattr(model, 'Ro', None))),
                base_lr
            )
            current_lr_ro = next(
                (pg['lr'] for pg in optimizer.param_groups
                 if len(pg['params']) == 1 and pg['params'][0] is getattr(model, 'Ro', None)),
                ro_base_lr
            )
        except Exception:
            current_lr_other, current_lr_ro = base_lr, ro_base_lr

        # 辅助函数：把秒数格式化为 H:MM:SS
        def _format_seconds(s):
            if s is None:
                return "N/A"
            s = int(s)
            m, sec = divmod(s, 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h:d}h{m:02d}m{sec:02d}s"
            if m > 0:
                return f"{m:d}m{sec:02d}s"
            return f"{sec:d}s"

        remaining = pbar.format_dict.get('remaining', None)
        eta_str = _format_seconds(remaining)

        pbar.set_postfix({
            'loss': f"{final_train_loss:.6f}",
            'Ro': f"{ro_history[-1]:.6f}",
            'lr_other': f"{current_lr_other:.2e}",
            'lr_Ro': f"{current_lr_ro:.2e}",
            'ETA': eta_str
        })
        pbar.refresh()

        if epoch % 1000 == 0:
            pbar.write("=" * 70)
            pbar.write(f"[Epoch {epoch}] 状态监控")
            pbar.write("=" * 70)
            pbar.write(f"  学习率: Ro: {current_lr_ro:.2e}, Other: {current_lr_other:.2e}")
            pbar.write(f"  Ro正则权重: {ro_weight:.4f}, Ro学习率缩放: {ro_lr_scale:.4f}")
            pbar.write(f"  θ mean: {theta.mean().item():.6f}, η mean: {eta.mean().item():.6f}")
            pbar.write(f"  u_pred mean: {u_pred.mean().item():.6f}, v_pred mean: {v_pred.mean().item():.6f}")
            pbar.write(f"  Z₀ mean: {Z0.mean().item():.6f}, Z₀ var: {Z0.var().item():.6f}")
            pbar.write(f"  Ro value: {model.Ro.item():.6f}")
            pbar.write(f"  Ro regularization: {ro_reg.item():.6f}")
            pbar.write(f"  B0_scalar: {model.B0_scalar.item():.6f}, B0_reg: {b0_reg.item():.6f}")
            pbar.write(f"  B1_scalar: {model.B1_scalar.item():.6f}, B1_reg: {b1_reg.item():.6f}")
            if epoch > 7000:
                pbar.write(f"  Ro change penalty: {ro_change_penalty.item():.6f}")
            pbar.write("损失分量:")
            pbar.write(f"    Data: {losses['data'].item():.6f}")
            pbar.write(f"    Dir:  {losses['dir'].item():.6f}")
            pbar.write(f"    Phys: {losses['phys'].item():.6f}")
            pbar.write(f"    Cont: {losses['cont'].item():.6f}")
            pbar.write(f"    Geo:  {losses['geo'].item():.6f}")
            pbar.write(f"    Act:  {losses['act'].item():.6f}")
            pbar.write(f"    Z0_reg: {z0_reg.item():.6f}")
            pbar.write(f"    Ro_reg: {ro_reg.item():.6f}")
            pbar.write(f"  Total Loss: {final_train_loss:.6f}")
            if hasattr(model, 'Ro') and model.Ro.grad is not None:
                pbar.write(f"  Ro grad: {model.Ro.grad.item():.6f}")
            pbar.write("=" * 70)

    # 保存训练结果
    pd.DataFrame({'Ro': ro_history}).to_csv(os.path.join(output_dir, 'ro_history.csv'), index=False)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_final.pt'))

    # 写 loss_history.csv（如果启用）
    if log_loss and loss_rows:
        pd.DataFrame(loss_rows).to_csv(loss_csv_path, index=False)
        print(f"✅ 已保存损失曲线数据到: {loss_csv_path}")

    print(f"✅ 模型训练完成，Ro 演化与参数已保存到: {output_dir}")

    summary = [
        "✅ 模型训练参数记录",
        f"Ro 初始值: {config.Ro}",
        f"Ro 最终值: {model.Ro.item():.6f}",
        f"omega_0: {config.omega_0}",
        f"velocity_scale: {config.velocity_scale}",
        f"grad_clip: {config.grad_clip}",
        f"训练轮数: {len(ro_history)}",
        "基础学习率: 1e-4",
        "Ro基准学习率: 5e-4",
        "损失权重:"
    ] + [f"  {k}: {v}" for k, v in loss_manager.weights.items()] + [
        "学习率策略: Ro学习率随正则权重同步衰减",
        "物理权重上限: 5.0",
        "Ro正则约束系数: 50"
    ]
    theta_abs = torch.mean(torch.abs(theta)).item()
    eta_abs = torch.mean(torch.abs(eta)).item()
    summary += [f"theta_abs: {theta_abs:.6f}", f"eta_abs: {eta_abs:.6f}"]

    with open(os.path.join(output_dir, "training_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    # 预测阶段（使用训练开始时已加载的 t_test/x_test/...）
    print("✅ 开始使用测试集进行预测...")

    (u_pred_test, v_pred_test, u_bar, v_bar, theta, eta, Z0,
     P, g1, g2, h1, h2, time_phase_C2, time_phase_C3) = compute_velocity(model, x_test, y_test, z_test, t_test)

    # 注意：测试阶段默认只做前向预测与误差评估，不强制计算需要 x/y 梯度连接的约束项（例如 geo）。
    # 这样可以避免在 requires_grad=False 或 no_grad 场景下出现 “未建立梯度连接” 报错。
    final_losses = {
        'data': value_loss(u_pred_test, v_pred_test, u_true_test, v_true_test),
        'dir': direction_loss(u_pred_test, v_pred_test, u_true_test, v_true_test),
        'act': activation_loss(theta, eta),
    }

    # 这些 residual 也可能依赖 autograd；若 compute_residuals 内部要求梯度连接，这里也做 try/except
    try:
        res_u_t, res_v_t, res_cont_t = compute_residuals(model, x_test, y_test, z_test, t_test)
        final_losses['phys'] = torch.mean(res_u_t ** 2 + res_v_t ** 2)
        final_losses['cont'] = torch.mean(res_cont_t ** 2)
    except Exception as e:
        print(f"⚠️  测试阶段跳过 phys/cont residual 计算（原因: {e}）")

    # geo 需要 Z0 对 x/y 的梯度连接；这里默认跳过
    # 如需启用，请确保 x_test/y_test.requires_grad_(True) 且不在 no_grad 作用域内计算 Z0

    true_mag = torch.sqrt(u_true_test ** 2 + v_true_test ** 2)
    mask = (true_mag > 0.005)
    pred_angle = torch.atan2(v_pred_test, u_pred_test)
    true_angle = torch.atan2(v_true_test, u_true_test)
    angle_diff = torch.abs(pred_angle - true_angle)
    angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)
    direction_mae_rad = angle_diff[mask].mean().item()
    final_losses['direction_mae'] = torch.tensor(direction_mae_rad)

    saver = PredictionSaver(model, scaler_mgr, output_dir)
    saver.save_all(
        original_df=original_test_df,
        u_pred=u_pred_test, v_pred=v_pred_test,
        u_true=u_true_test, v_true=v_true_test,
        Z0=Z0, x=x_test, y=y_test, z=z_test, t=t_test,
        g1=g1, g2=g2,
        final_losses=final_losses,
        extra_metrics={'Ro': model.Ro.item()}
    )
    print("✅ 测试集预测完成，结果已保存到:", output_dir)

    # 后续绘图与保存保持不变（略）
    ...


if __name__ == "__main__":
    args = parse_args_for_train()
    train_prediction_model(train_path=args.train, test_path=args.test, input_path=args.input,
                           output_dir=args.output, device=args.device,
                           log_loss=args.log_loss, log_loss_every=args.log_loss_every)
