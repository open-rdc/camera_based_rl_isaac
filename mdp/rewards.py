# isaaclab_tasks/my_custom_task/reward.py

from __future__ import annotations

import torch
import torch.nn.functional as F

from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils.math import euler_xyz_from_quat

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
    reward += reached.float() * 30 * lin_vel_mag
    curr_wp_idx[reached] = (curr_wp_idx[reached] + 1) % num_wps

    return reward

def slip_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # ---- パラメータ（ロボット特性） ----
    wheel_radius = 0.2
    wheel_base = 0.8
    # --------------------------------

    action = env.action_manager.action
    state = env.scene.get_state()

    root_pose = state["articulation"][asset_cfg.name]["root_pose"]       # [N, 7]
    root_vel = state["articulation"][asset_cfg.name]["root_velocity"]    # [N, 6] (vx, vy, vz, wx, wy, wz)

    lin_vel_xy = root_vel[:, 0:2]  # [vx, vy]
    ang_vel_z = root_vel[:, 5]     # wz

    # ロボット向き（yaw）をクォータニオンから取得
    roll, pitch, yaw  = euler_xyz_from_quat(root_pose[:, 3:7])  # [N]

    left_vel = action[:, 0]
    right_vel = action[:, 1]

    v = wheel_radius * (right_vel + left_vel) / 2
    omega = wheel_radius * (right_vel - left_vel) / wheel_base

    ideal_vx = v * torch.cos(yaw)  # [N]
    ideal_vy = v * torch.sin(yaw)  # [N]

    ideal_vel_xy = torch.stack([ideal_vx, ideal_vy], dim=1)

    # omega と ang_vel_z も reshape
    omega = omega.view(-1, 1)
    ang_vel_z = ang_vel_z.view(-1, 1)

    ideal_vec = F.normalize(torch.cat([ideal_vel_xy, omega], dim=1), dim=1)
    actual_vec = F.normalize(torch.cat([lin_vel_xy, ang_vel_z], dim=1), dim=1)

    cos_sim = torch.sum(actual_vec * ideal_vec, dim=1)

    penalty = -1.0 * (1.0 - cos_sim)

    return penalty


def go_ahead(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    state = env.scene.get_state()

    joint_velocities = state["articulation"][asset_cfg.name]["joint_velocity"]

    joint_names = asset.data.joint_names
    left_idx = joint_names.index("left_wheel_joint")
    right_idx = joint_names.index("right_wheel_joint")

    left_speed = joint_velocities[:, left_idx]
    right_speed = joint_velocities[:, right_idx]

    # 前進速度（後退は報酬ゼロ）
    forward_speed = 0.5 * (left_speed + right_speed)
    reward = torch.clamp(forward_speed, min=0.0)

    return reward
