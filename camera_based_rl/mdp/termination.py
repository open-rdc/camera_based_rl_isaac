from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

import cv2
from isaaclab.utils import convert_dict_to_backend

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from rewards import compute_reward

def termination(self, action):
    # 環境のステップ処理
    obs, reward, done, info = self.env.step(action)

    # 報酬の再計算
    reward = compute_reward(obs, action, next_obs, info)

    return obs, reward, done, info