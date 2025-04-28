# src/models/cnn_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraPolicyNetwork(nn.Module):
    def __init__(self, output_dim=128):
        """
        Args:
            output_dim (int): 最後に出力する潜在特徴ベクトルの次元数
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2)  # (480x300) -> (120x75)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # (120x75) -> (60x38)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # (60x38) -> (30x19)

        # 計算：30x19x128
        self.fc = nn.Linear(30 * 19 * 128, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, 480, 300) RGB画像
        Returns:
            (batch_size, output_dim) 特徴ベクトル
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
