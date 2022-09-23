"""
Wrappers for VGDL Games
"""
import numpy as np

# import timeout_decorator

import gym

import a2c_ppo_acktr.envs as torch_env

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from .env import PCGEnv


def make_env(seed, rank, log_dir, allow_early_resets, **env_kwargs):
    def _thunk():
        env = NormalizeObservation(ChannelFirstObservation(PCGEnv()))
        return env
    return _thunk


def make_vec_envs(
    seed,
    num_processes,
    log_dir,
    device,
    allow_early_resets,
    num_frame_stack=None,
):
    envs = [
        make_env(seed=seed, rank=i, log_dir=log_dir,
                 allow_early_resets=allow_early_resets)
        for i in range(num_processes)
    ]
    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    envs = torch_env.VecPyTorch(envs, device)
    return envs


class ChannelFirstObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[2], shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=shape, dtype=np.float32
        )

    def observation(self, observation):
        return observation/255.0
