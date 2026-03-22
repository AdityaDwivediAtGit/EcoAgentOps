"""Minimal Gym environment scaffold for pruning actions.

State: proxy utilities, proxy energy, carbon, batch index
Actions: 0=prune, 1=keep, 2=compress

This is a template — adapt reward and state generation for offline batch training.
"""
import gym
from gym import spaces
import numpy as np


class PruningEnv(gym.Env):
    def __init__(self, batch_size=32):
        super().__init__()
        # observation: u, e_proxy, carbon
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        # create a random batch stat
        self.u = np.random.rand()
        self.e = np.random.rand() * 0.1
        self.carbon = 0.72
        obs = np.array([self.u, self.e, self.carbon], dtype=np.float32)
        return obs

    def step(self, action):
        # Simplified reward: keep accuracy - lambda * energy * carbon
        acc_gain = (1 - self.u) * (0.1 if action == 1 else 0.05)
        energy = self.e * (0.5 if action == 0 else 1.0)
        reward = acc_gain - 0.01 * energy * self.carbon
        done = True  # one-step episodic for batch decisions
        obs = np.array([np.random.rand(), np.random.rand() * 0.1, self.carbon], dtype=np.float32)
        info = {"acc_gain": acc_gain, "energy": energy}
        return obs, float(reward), done, info


if __name__ == "__main__":
    env = PruningEnv()
    o = env.reset()
    print("sample obs:", o)
    print(env.step(1))
