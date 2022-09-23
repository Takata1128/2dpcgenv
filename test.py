from rl.env import PCGEnv
from rl.wrappers import ChannelFirstObservation
from rl.ppo.policy import Policy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
env = ChannelFirstObservation(PCGEnv())

obs = env.reset()
plt.imshow(obs.transpose(1, 2, 0))
# cv2.imshow('image',obs)
actor_critic = Policy(
    env.observation_space.shape,
    env.action_space,
    base_kwargs={"recurrent": True},
    # activation='sigmoid'
)
actor_critic.load_state_dict(torch.load(
    "/root/mnt/2dpcgenv/checkpoints/ppo/2D PCG_20220917072842.pt"))

rnn_hxs = torch.zeros((1, actor_critic.recurrent_hidden_state_size))


done = False
while not done:
    obs = torch.tensor(obs).unsqueeze(0)
    value, action, action_log_probs, rnn_hxs = actor_critic.act(
        obs, rnn_hxs, masks=1.0, deterministic=False)
    obs, reward, done, _ = env.step(action[0].detach().numpy())
    print(env.blocks)
    print("Action", action)
    print("reward:", reward)
    print("Done:", done)
    plt.imshow(obs.transpose(1, 2, 0))
