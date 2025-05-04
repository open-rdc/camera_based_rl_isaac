from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

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

    # 速度の絶対値の平均を報酬とする（前進速度に比例）
    left_speed = torch.abs(joint_velocities[:, left_idx])
    right_speed = torch.abs(joint_velocities[:, right_idx])
    reward = 0.5 * (left_speed + right_speed)

    return reward