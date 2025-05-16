import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Network(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        image_space = observation_space.spaces["image"]
        self.image_shape = image_space.shape  # (H, W, C)

        n_input_channels = self.image_shape[2]  # C=3 for RGB

        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        # 出力サイズを計算
        with torch.no_grad():
            sample_input = torch.zeros(1, *self.image_shape).permute(0, 3, 1, 2)  # (1, C, H, W)
            x = self.flatten(
                self.relu(self.conv3(
                    self.relu(self.conv2(
                        self.relu(self.conv1(sample_input))
                    ))
                ))
            )
            conv_output_size = x.shape[1]

        self.fc4 = nn.Linear(conv_output_size, 512)
        self.fc5 = nn.Linear(512, features_dim)

        # 重み初期化
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)

        self.cnn_layer = nn.Sequential(
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.flatten
        )
        self.fc_layer = nn.Sequential(
            self.fc4, self.relu,
            self.fc5
        )

        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        x = observations["image"]  # shape: (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # shape: (B, C, H, W)
        x = self.cnn_layer(x)
        return self.fc_layer(x)