import dataclasses


@dataclasses.dataclass
class DreamerConfig:
    project: str = 'Learning 2DPCG using Dreamer'

    device: str = 'cuda'
    buffer_capacity: int = 200000

    # state_dim
    state_dim: int = 30
    rnn_hidden_dim: int = 200

    # learning rate and epsilon
    model_lr: float = 6e-4
    value_lr: float = 8e-5
    action_lr: float = 8e-5
    eps: float = 1e-4

    # other hyper params
    seed_episodes: int = 100
    all_episodes: int = 1000
    test_interval: int = 50
    model_save_interval: int = 200
    collect_interval: int = 50

    action_noise_var: float = 0.05

    batch_size: int = 50
    chunk_length: int = 50
    imagination_horizon: int = 15

    gamma: float = 0.95
    lambda_: float = 0.95
    clip_grad_norm: float = 100.0
    free_nats: float = 3.0
