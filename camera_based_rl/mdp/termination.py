from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import convert_dict_to_backend

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from shapely.geometry import LineString, Point

## episode end condition
def out_of_course_area(env: ManagerBasedRLEnv, center_line: list[tuple[float, float]], course_width: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Args:
        env: IsaacLab 環境
        center_line: コース中心を構成する座標列（白線）
        course_width: コースの幅（左右に course_width/2 バッファを持たせる）
        asset_cfg: チェック対象となるエージェント名など（SceneEntityCfg）
    """
    center_path = LineString(center_line)
    buffered_area = center_path.buffer(course_width / 2.0)

    state = env.scene.get_state()
    root_pose = state["articulation"][asset_cfg.name]["root_pose"]
    robot_xy = root_pose[:, :2]

    results = []
    for xy in robot_xy.tolist():
        pt = Point(xy[0], xy[1])
        results.append(not buffered_area.contains(pt))

    return torch.tensor(results, dtype=torch.bool, device=env.device)