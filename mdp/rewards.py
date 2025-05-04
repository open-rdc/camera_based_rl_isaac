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
    
def compute_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward is proportional to the average absolute wheel speed."""

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
    reward = torch.clamp(forward_speed, min=0.0)  # shape: [num_envs]

    return reward