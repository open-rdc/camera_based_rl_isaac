# isaaclab_tasks/my_custom_task/reward.py

import torch

def compute_reward(obs, action, next_obs, info):
    """
    Args:
        obs (torch.Tensor): 現在の観測状態。
        action (torch.Tensor): 実行されたアクション。
        next_obs (torch.Tensor): 次の観測状態。
        info (dict): 環境からの追加情報。
    
    Returns:
        torch.Tensor: 計算された報酬。
    """
    # 速度に関する報酬
    speed = next_obs["linear_velocity"]  # 速度ベクトル
    speed_reward = torch.norm(speed, dim=-1)  # 速度の大きさ

    # 進行方向に関する報酬
    heading = next_obs["heading_vector"]  # 進行方向ベクトル
    track_direction = next_obs["track_direction"]  # トラックの方向ベクトル
    heading_alignment = torch.sum(heading * track_direction, dim=-1)  # コサイン類似度
    heading_reward = heading_alignment

    # トラックからの逸脱に対するペナルティ
    off_track = next_obs["off_track"]  # トラックからの逸脱フラグ（0または1）
    off_track_penalty = -1.0 * off_track

    # 衝突に対するペナルティ
    collision = next_obs["collision"]  # 衝突フラグ（0または1）
    collision_penalty = -1.0 * collision

    # 総合報酬の計算
    reward = speed_reward + heading_reward + off_track_penalty + collision_penalty

    return reward
