###################################################
######### Agent Interface for Evaluation ##########
###################################################

"""
This file will be used during the evaluation of your code.
Define ONLY: convert_obs, convert_action, and Agent.
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def convert_obs(obs: np.ndarray) -> torch.Tensor:
    """
    Convert raw env observation into NN-friendly format.

    - Car state: first 16 values (unchanged)
    - Cones: [used, dist, sin(theta), cos(theta), sideL, sideR] → 6 values per cone
    - Centerline: [dist, sin(dir_angle), cos(dir_angle), sin(tangent), cos(tangent)] → 5 values per point
    - After concatenation, drop indices [4..10] (position, RGB, distance)
    """
    src = np.asarray(obs, dtype=np.float32).copy()

    # --- car state ---
    car_state = src[0:16].astype(np.float32)

    # --- cones (80 cones × 5 raw values) ---
    cones_out = []
    for i in range(80):
        base = 16 + i * 5
        used = float(src[base])
        x, y = float(src[base + 1]), float(src[base + 2])
        sideL, sideR = float(src[base + 3]), float(src[base + 4])

        if used == 1.0:
            dist = np.sqrt(x * x + y * y)
            ang = np.arctan2(y, x)
            s, c = np.sin(ang), np.cos(ang)
        else:
            dist, s, c = 0.0, 0.0, 0.0

        cones_out.extend([used, dist, s, c, sideL, sideR])
    cones_out = np.asarray(cones_out, dtype=np.float32)

    # --- centerline (20 points × 2 raw values) ---
    center_out = []
    for i in range(20):
        base = 416 + i * 2
        x, y = float(src[base]), float(src[base + 1])

        dist = np.sqrt(x * x + y * y)
        a = np.arctan2(y, x)
        s_dir, c_dir = np.sin(a), np.cos(a)

        if i == 0:
            s_tan, c_tan = 0.0, 0.0
        else:
            dy, dx = src[base + 1] - src[base - 1], src[base] - src[base - 2]
            t = np.arctan2(dy, dx)
            s_tan, c_tan = np.sin(t), np.cos(t)

        center_out.extend([dist, s_dir, c_dir, s_tan, c_tan])
    center_out = np.asarray(center_out, dtype=np.float32)

    # --- final feature vector ---
    final = np.concatenate([car_state, center_out, cones_out], axis=0)

    # drop indices [4..10] (position x,y, yaw angle, RGB, distance)
    converted = np.delete(final, list(range(4, 11)))

    return torch.from_numpy(converted).float()


def convert_action(action: torch.Tensor) -> np.ndarray:
    """
    Convert NN action → environment action.
    Ensures shape (3,) and clips to [-1, 1].
    """
    arr = action.detach().cpu().numpy()
    arr = np.atleast_2d(arr)
    arr = np.clip(arr, -1.0, 1.0)
    if arr.shape[0] == 1:
        return arr[0]
    return arr


class Agent(nn.Module):
    """
    Minimal MLP actor:
      - fc1: obs_dim -> 64, ReLU
      - fc2: 64 -> 128, ReLU
      - fc_mu: 128 -> act_dim, tanh
    """

    def __init__(self, env):
        super().__init__()
        action_shape = env.action_space.shape
        obs_dim = int(convert_obs(env.reset()[0]).shape[0])
        act_dim = int(np.prod(action_shape))

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, act_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)
