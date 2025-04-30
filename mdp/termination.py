from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

import cv2
from isaaclab.utils import convert_dict_to_backend

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from mdp.reward import compute_reward

## episode end condition
def termination(env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the robot's x position exceeds the given bounds."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the robot's current x position
    pos = asset.data.root_pos_w[..., 0]  # x-axis

    # Terminate if x is out of bounds
    return (pos < bounds[0]) | (pos > bounds[1])