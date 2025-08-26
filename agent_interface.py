########################################################
#   Evaluation Interface for Racing Agent
########################################################
"""
Must expose:
- convert_obs(np.ndarray) -> torch.Tensor
- convert_action(torch.Tensor) -> np.ndarray
- Agent(env) : nn.Module with get_action(obs)
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ==========================
# Observation converter
# ==========================
def convert_obs(obs: np.ndarray) -> torch.Tensor:
    raw = np.asarray(obs, dtype=np.float32).copy()

    # car state (first 16)
    car_feats = raw[0:16].astype(np.float32)

    # cones: [used, dist, sin, cos, sideL, sideR] for 80 cones
    cone_feats = []
    for i in range(80):
        idx = 16 + 5 * i
        used, x, y, sideL, sideR = raw[idx: idx + 5]
        if used == 1.0:
            d = np.sqrt(x ** 2 + y ** 2)
            ang = np.arctan2(y, x)
            s, c = np.sin(ang), np.cos(ang)
        else:
            d, s, c = 0.0, 0.0, 0.0
        cone_feats.extend([used, d, s, c, sideL, sideR])
    cone_feats = np.asarray(cone_feats, dtype=np.float32)

    # centerline: 20 pts -> [dist, sin(dir), cos(dir), sin(tan), cos(tan)]
    cl_feats = []
    for i in range(20):
        base = 416 + 2 * i
        x, y = raw[base: base + 2]
        d = np.sqrt(x ** 2 + y ** 2)
        ang = np.arctan2(y, x)
        s_dir, c_dir = np.sin(ang), np.cos(ang)
        if i == 0:
            s_t, c_t = 0.0, 0.0
        else:
            dy, dx = raw[base + 1] - raw[base - 1], raw[base] - raw[base - 2]
            t = np.arctan2(dy, dx)
            s_t, c_t = np.sin(t), np.cos(t)
        cl_feats.extend([d, s_dir, c_dir, s_t, c_t])
    cl_feats = np.asarray(cl_feats, dtype=np.float32)

    # final vector and small cleanup (drop [4..10])
    vec = np.concatenate([car_feats, cl_feats, cone_feats], axis=0)
    cleaned = np.delete(vec, list(range(4, 11)))
    return torch.from_numpy(cleaned).float()


# ==========================
# Action converter
# ==========================
def convert_action(action: torch.Tensor) -> np.ndarray:
    arr = action.detach().cpu().numpy()
    arr = np.atleast_2d(arr)
    arr = np.clip(arr, -1.0, 1.0)
    return arr[0] if arr.shape[0] == 1 else arr


# ==========================
# Actor (bigger MLP: 256-256)
# ==========================
class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(convert_obs(env.reset()[0]).shape[0])
        act_dim = int(np.prod(env.action_space.shape))

        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, act_dim)

        # rescale to env bounds (assumed symmetric [-1,1])
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.mu(x))
        return x * self.action_scale + self.action_bias

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)
