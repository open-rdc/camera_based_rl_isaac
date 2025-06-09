#!/usr/bin/env python3
import sys
import os
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from stable_baselines3 import PPO
from cnn import Network


class ImageToCmdVelNode(Node):
    def __init__(self, model_path):
        super().__init__('image_to_cmdvel_node')

        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ モデルファイルが存在しません: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.get_logger().info(f"✅ モデル読み込み: {model_path}")
        self.model = PPO.load(model_path, custom_objects={"features_extractor_class": Network})

        self.model.policy.eval()

        self.image_size = (200, 88)
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    def preprocess_image(self, cv_image):
        resized = cv2.resize(cv_image, self.image_size)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0)
        return array

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            obs = self.preprocess_image(cv_image)
            print(f"obs dtype :{obs.dtype}")
            print(f"obs type :{type(obs)}")
            print(f"obs shape :{obs.shape}")

            with torch.no_grad():
                action, _ = self.model.predict(obs, deterministic=True)

            if isinstance(action, (np.ndarray, list, torch.Tensor)):
                action = np.array(action).squeeze()
                if action.ndim != 1 or action.shape[0] != 2:
                    raise ValueError(f"期待されるアクション形式は shape=(2,) ですが、受け取ったのは: {action.shape}")
                v_l, v_r = float(action[0]), float(action[1])
            else:
                raise ValueError(f"予期しないアクションの型: {type(action)}")


            r = 0.2   # タイヤ半径 [m]
            L = 0.7   # トレッド [m]

            v = 0.5 * r * (v_r + v_l)
            w = r / L * (v_r - v_l)

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w

            self.publisher.publish(cmd)
            self.get_logger().info(f"🌀 左: {v_l:.2f}, 右: {v_r:.2f} → 並進: {v:.2f} m/s, 角速度: {w:.2f} rad/s")
        except Exception as e:
            self.get_logger().error(f"画像処理エラー: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test.py /path/to/model.zip")
        sys.exit(1)

    model_path = sys.argv[1]

    rclpy.init()
    node = ImageToCmdVelNode(model_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
