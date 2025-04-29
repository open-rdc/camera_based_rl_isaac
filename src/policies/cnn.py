# src/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CameraPolicyFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        """
        Args:
            observation_space (gym.Space): 環境の観測空間
            features_dim (int): 出力する特徴ベクトルの次元数
        """
        super().__init__(observation_space, features_dim)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Flatten後のサイズを計算
        self._n_flatten = self._get_flattened_size(observation_space)

        self.fc = nn.Linear(self._n_flatten, features_dim)

    def _get_flattened_size(self, observation_space):
        with torch.no_grad():
            # observation_spaceのshapeは(H, W, C)なので、(C, H, W)に変換
            sample_input = torch.zeros(1, *observation_space.shape)
            sample_input = sample_input.permute(0, 3, 1, 2)
            x = F.relu(self.conv1(sample_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            n_flatten = x.view(1, -1).shape[1]
        return n_flatten

    def forward(self, observations):
        # (batch, H, W, C) -> (batch, C, H, W)
        x = observations.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
