import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Network(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        if isinstance(observation_space, gym.spaces.Dict):
            image_space = observation_space.spaces["image"]
        else:
            image_space = observation_space

        image_shape = image_space.shape  # (H, W, C)
        n_input_channels = image_shape[2]

        super().__init__(observation_space, features_dim)

        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            sample_input = torch.zeros(1, *image_shape).permute(0, 3, 1, 2)  # (1, C, H, W)
            conv_out = self.conv3(
                self.relu(self.conv2(
                    self.relu(self.conv1(sample_input))
                ))
            )
            conv_output_size = conv_out.reshape(1, -1).shape[1]

        self.fc4 = nn.Linear(conv_output_size, 512)
        self.fc5 = nn.Linear(512, features_dim)

        # 重み初期化
        for layer in [self.conv1, self.conv2, self.conv3, self.fc4]:
            nn.init.kaiming_normal_(layer.weight)

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

    def forward(self, observations) -> torch.Tensor:
        if isinstance(observations, dict):
            x = observations["image"]
        else:
            x = observations  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.cnn_layer(x)
        return self.fc_layer(x)
