# isaaclab_tasks/my_custom_task/reward.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

import cv2
from math import sqrt
from isaaclab.utils import convert_dict_to_backend
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# グローバルまたはモジュールスコープのインデックス
_wp_idx = None

def get_wp_idx(num_envs: int, device: torch.device) -> torch.Tensor:
    global _wp_idx
    if _wp_idx is None or _wp_idx.shape[0] != num_envs:
        _wp_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
    return _wp_idx

def reset_wp_idx(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    global _wp_idx
    """Reset waypoint indices for each parallel environment."""
    _wp_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

def target_path_reward(env: ManagerBasedRLEnv, waypoints: list[tuple[float, float]], asset_cfg: SceneEntityCfg, radius: float = 5.0) -> torch.Tensor:
    device = env.device
    num_envs = env.num_envs
    wp_tensor = torch.tensor(waypoints, dtype=torch.float32, device=device)
    num_wps = wp_tensor.shape[0]

    # 各環境ごとのインデックス
    curr_wp_idx = get_wp_idx(num_envs, device)

    state = env.scene.get_state()
    root_pose = state["articulation"][asset_cfg.name]["root_pose"]
    root_vel = state["articulation"][asset_cfg.name]["root_velocity"]

    robot_xy = root_pose[:, :2]
    lin_vel_xy = root_vel[:, :2]

    origin_dists = robot_xy.norm(dim=1)
    reset_mask = origin_dists < 5.0
    if reset_mask.any():
        reset_wp_idx(env, asset_cfg)

    to_goal_vec = wp_tensor[curr_wp_idx] - robot_xy
    to_goal_unit = to_goal_vec / (to_goal_vec.norm(dim=1, keepdim=True) + 1e-6)
    lin_vel_mag = lin_vel_xy.norm(dim=1)
    lin_vel_unit = lin_vel_xy / (lin_vel_mag.unsqueeze(1) + 1e-6)

    alignment = torch.sum(lin_vel_unit * to_goal_unit, dim=1)
    reward = lin_vel_mag * alignment.clamp(min=0.0)

    dists = to_goal_vec.norm(dim=1)
    reached = dists < radius
    reward += reached.float() * 300.0
    curr_wp_idx[reached] = (curr_wp_idx[reached] + 1) % num_wps

    return reward