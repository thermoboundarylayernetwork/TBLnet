class ModelConfig:
    def __init__(
        self,
        Ro=0.024,
        omega_0=30.0,
        use_scaler=True,
        scaler_mgr=None,
        depth_scaler=None,
        siren_layers=None,
        velocity_scale=1.0,
        grad_clip=10.0
    ):
        """
        模型配置类，用于集中管理所有可调参数。

        参数说明：
        - Ro: Rossby 数（控制非量纲展开）
        - omega_0: SIREN 激活频率
        - use_scaler: 是否使用归一化器
        - scaler_mgr: 全局归一化器管理器
        - depth_scaler: 专用于深度的归一化器
        - siren_layers: SIREN 网络结构，例如 [4, 128, 128, 128, 128, 128, 15]
        - velocity_scale: 非量纲速度恢复为物理单位的比例（如 m/s）
        - grad_clip: 自动裁剪导数的阈值，防止梯度爆破
        """
        self.Ro = Ro
        self.omega_0 = omega_0
        self.use_scaler = use_scaler
        self.scaler_mgr = scaler_mgr
        self.depth_scaler = depth_scaler
        self.siren_layers = siren_layers or [4, 128, 128, 128, 128, 128, 15]
        self.velocity_scale = velocity_scale
        self.grad_clip = grad_clip
