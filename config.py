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
        Model configuration class for managing all tunable hyperparameters.

        Args:
            Ro: Rossby number (controls nondimensional expansion)
            omega_0: SIREN activation frequency
            use_scaler: Whether to use feature normalization
            scaler_mgr: Global normalization manager
            depth_scaler: Normalizer for depth feature
            siren_layers: SIREN neural network structure, e.g. [4, 128, 128, 128, 128, 128, 15]
            velocity_scale: Scale to recover dimensional velocity (e.g., m/s)
            grad_clip: Gradient clipping threshold to prevent explosion
        """
        self.Ro = Ro
        self.omega_0 = omega_0
        self.use_scaler = use_scaler
        self.scaler_mgr = scaler_mgr
        self.depth_scaler = depth_scaler
        self.siren_layers = siren_layers or [4, 128, 128, 128, 128, 128, 15]
        self.velocity_scale = velocity_scale
        self.grad_clip = grad_clip