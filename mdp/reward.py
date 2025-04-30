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
    """Penalize camera position deviation from a target value."""

    asset: Articulation = env.scene[asset_cfg.name]

    single_cam_data = convert_dict_to_backend(
                {k: v[0] for k, v in asset.data.output.items()}, backend="numpy")

    # compute the reward
    img_gray = cv2.cvtColor(single_cam_data['rgb'], cv2.COLOR_BGR2GRAY)
    thr, img_th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    # Find contours of the black regions
    contours, _ = cv2.findContours(cv2.bitwise_not(img_th), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize variables to store the center coordinates
    centers = []
    # Loop through contours to calculate centroids
    for contour in contours:
        # Calculate moments
        M = cv2.moments(contour)
        if M['m00'] != 0:
            # Calculate x, y coordinates of the centroid
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centers.append((cX, cY))

    if len(centers) > 0:
        normilized_dist = 1 - sqrt((160 - centers[0][0])**2 + (120 - centers[0][1])**2)/200 
    else:
        normilized_dist = 0

    return normilized_dist
