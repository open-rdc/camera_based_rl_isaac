# src/train.py

import os
import sys
import random
import yaml
from datetime import datetime

import gymnasium as gym
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

from src.models.camera_feature_extractor import CameraPolicyFeaturesExtractor
from isaaclab.app import AppLauncher
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.envs import multi_agent_to_single_agent, DirectMARLEnv
import isaaclab_tasks  # 必要: タスク登録

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "sb3_config.yaml"))
config = load_config(config_path)

# --- 1. Isaac Simを立ち上げる ---
args_cli, _ = AppLauncher.default_argparser().parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. 環境を作成 ---
TASK_NAME = config.get("task_name", "Isaac-Cartpole-v0")

env = gym.make(TASK_NAME, render_mode=None)

if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

env = Sb3VecEnvWrapper(env)

if config.get("use_vecnormalize", True):
    env = VecNormalize(
        env,
        training=True,
        norm_obs=config.get("normalize_obs", True),
        norm_reward=config.get("normalize_reward", True),
        clip_obs=config.get("clip_obs", 10.0),
        gamma=config.get("gamma", 0.99),
        clip_reward=np.inf,
    )

# --- 3. Agentの設定 ---
policy_kwargs = {
    "features_extractor_class": CameraPolicyFeaturesExtractor,
    "features_extractor_kwargs": {"features_dim": config.get("features_dim", 128)},
}

agent = PPO(
    policy="CnnPolicy",
    env=env,
    tensorboard_log="./logs/tb/",
    verbose=1,
    policy_kwargs=policy_kwargs,
    **config["ppo_params"]
)

# --- 4. ログ・チェックポイント設定 ---
log_root_path = os.path.abspath(os.path.join("logs", "sb3", TASK_NAME))
run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(log_root_path, run_info)

os.makedirs(log_dir, exist_ok=True)

new_logger = configure(log_dir, ["stdout", "tensorboard"])
agent.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
    save_freq=config.get("checkpoint_save_freq", 10000),
    save_path=log_dir,
    name_prefix="model",
    verbose=2,
)

# --- 5. 学習 ---
total_timesteps = int(config.get("total_timesteps", 1e6))
agent.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# --- 6. モデル保存 & 終了 ---
agent.save(os.path.join(log_dir, "final_model"))

env.close()
simulation_app.close()
