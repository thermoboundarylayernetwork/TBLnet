#这部分是附件之一：对于运行的设备、输出的残差；损失的权重等，构造一致的

import torch

def get_device():
    if torch.cuda.is_available():
        print("training with cuda")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("training with mps")
        return torch.device('mps')
    else:
        print("training with cpu")
        return torch.device('cpu')

def rescale_to_physical_units(u_pred, v_pred, velocity_scale=1.0):
    """
    将预测速度从标准化单位转换为物理单位（如 m/s）
    """
    u_out = u_pred * velocity_scale
    v_out = v_pred * velocity_scale
    return u_out, v_out

def prepare_inputs(x, y, z, t):
    """
    自动设置输入张量的 requires_grad 属性，确保支持 autograd 梯度计算。
    返回处理后的张量。
    """
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    return x, y, z, t



def validate_residual_consistency(u_true, u_pred, v_true, v_pred, output_dir):
    """
    自动检查残差分布与散点图趋势是否一致
    输出诊断报告并保存图像
    """
    def analyze(name, true, pred):
        true = true.detach().cpu().flatten()
        pred = pred.detach().cpu().flatten()
        residual = pred - true

        mean_residual = residual.mean().item()
        skewness = ((residual - residual.mean())**3).mean() / (residual.std()**3 + 1e-8)
        bias_direction = (pred < true).float().mean().item()  # 比例偏低

        # 散点图
        plt.figure(figsize=(5, 5))
        plt.scatter(true.numpy(), pred.numpy(), alpha=0.3, s=10)
        plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--')
        plt.xlabel(f"{name}_true")
        plt.ylabel(f"{name}_pred")
        plt.title(f"{name}: Predicted vs True")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_scatter_check.png")
        plt.close()

        # 残差直方图
        plt.figure(figsize=(5, 4))
        plt.hist(residual.numpy(), bins=50, color='lightgreen', edgecolor='black')
        plt.title(f"{name} Residual Histogram")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_residual_hist_check.png")
        plt.close()

        # 文本报告
        return {
            'mean_residual': mean_residual,
            'skewness': skewness.item(),
            'bias_direction_ratio': bias_direction
        }

    u_report = analyze("u", u_true, u_pred)
    v_report = analyze("v", v_true, v_pred)

    # 保存诊断报告
    with open(f"{output_dir}/residual_consistency_report.txt", 'w') as f:
        for name, report in zip(["u", "v"], [u_report, v_report]):
            f.write(f"{name}:\n")
            for k, v in report.items():
                f.write(f"  {k}: {v:.6f}\n")
            f.write("\n")


class LossManager:
    def __init__(self, weights=None):
        self.weights = weights or {
            'data': 1.0,
            'dir': 1.0,
            'phys': 1.0,
            'cont': 1.0,
            'geo': 1.0,
            'act': 0.01,
            'var': 0.01
        }
        self.history = []

    def compute_total_loss(self, losses: dict, epoch: int = None):
        total = sum(self.weights[key] * losses[key] for key in self.weights)
        self.history.append({
            'epoch': epoch,
            'total_loss': total.item(),
            **{f"{k}_loss": losses[k].item() for k in losses}
        })
        return total

    def get_history(self):
        return self.history

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights: dict):
        self.weights.update(new_weights)

