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
    
def time_loss_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    current_step = env.episode_length_s  # shape: [num_envs]

    max_steps = env.max_episode_length

    # 正規化されたステップ（0〜1）を使って報酬減少（0〜1）
    # 例: ステップ数が増えるほど reward が減る
    decay = current_step / max_steps  # shape: [num_envs]

    reward = -decay  # shape: [num_envs]

    return reward