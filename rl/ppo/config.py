from dataclasses import dataclass


@dataclass
class Config:
    project: str = "2D PCG Reinforcement Training"
    env_name: str = "2D PCG"

    model_save_dir: str = "/root/mnt/2dpcgenv/checkpoints"
    log_dir: str = "log"
    cuda: bool = True
    num_steps: int = 32
    use_gae: bool = True
    gae_lambda: float = 0.95
    use_proper_time_limits: bool = False
    seed: int = 0
    algo_name: str = "ppo"
    checkpoint = None
    num_processes: int = 32
    gamma: float = 0.99
    recurrent_policy: bool = True
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 2.5e-4
    eps: float = 1e-5
    max_grad_norm: float = 0.5
    clip_param: float = 0.1
    ppo_epoch: int = 4
    num_mini_batch: int = 4
    alpha: float = 0.99

    save_interval: int = 100
    log_interval: int = 10

    steps: int = int(1e7)
