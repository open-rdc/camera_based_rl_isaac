# isaaclab_tasks/my_custom_task/mdp.py

from isaaclab.envs import ManagerBasedRLEnv
import torch

def reset_wp_idx(env: ManagerBasedRLEnv):
    """Reset waypoint indices for each parallel environment."""
    global _wp_idx
    _wp_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)