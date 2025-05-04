# camera_baed_rl_isaac

## Overview
本プロジェクトは、Isaac Lab を基盤とした強化学習環境で、RGB画像を観測として使用し、並列に複数エージェントを学習させるためのシステムです。
カスタムタスク camera_based_rl において、画像入力から特徴ベクトルを抽出するCNNベースのネットワークを使用し、画像からのポリシー学習が可能です。

## Example

1. demo
デモ用のコード実行（3つの並列環境を作成）
```
python create_simulator_env.py --enable_cameras --num_envs 3
```

2. training
フォルダをIsaac Lab上に移動させます


training用のコードを実行し, タスクを与えることで訓練が行えます.
```
cd ~/IsaacLab/scripts/reinforcement_learning/sb3
python train.py --task camera_based_rl --enable_cameras --num_envs 3
```
