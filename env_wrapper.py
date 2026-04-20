# ─── env_wrapper.py ────────────────────────────────────────
import numpy as np
import gymnasium as gym
import gymnasium_robotics

ENV_ID = "HandManipulateBlockRotateXYZ-v1"


def quat_distance(q1, q2):
    """Angular distance between two quaternions (radians)."""
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


class DenseRewardWrapper(gym.Wrapper):
    """
    Dense reward:
      r = -rotation_error - 0.1 * position_error + 10 * success_bonus
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        ag = obs["achieved_goal"]    
        dg = obs["desired_goal"]    

        pos_err  = np.linalg.norm(ag[:3] - dg[:3])
        quat_err = quat_distance(ag[3:], dg[3:])

        reward = -quat_err - 0.1 * pos_err

        if info.get("is_success", False):
            reward += 10.0
            terminated = True

        return obs, reward, terminated, truncated, info


class SparseRewardWrapper(gym.Wrapper):
    """
    Sparse reward: 0 on success, -1 otherwise.
    Best used with HER.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("is_success", False):
            terminated = True
        return obs, reward, terminated, truncated, info


def make_env(reward_type="dense", render_mode=None):
    """Factory — returns the correctly wrapped environment."""
    env = gym.make(ENV_ID, render_mode=render_mode)
    if reward_type == "dense":
        return DenseRewardWrapper(env)
    return SparseRewardWrapper(env)