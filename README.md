# camera_based_rl_isaac

## Overview
本プロジェクトは、Isaac Lab を基盤とした強化学習環境で、RGB画像を観測として使用し、並列に複数エージェントを学習させるためのシステムです。
カスタムタスク camera_based_rl において、画像入力から特徴ベクトルを抽出するCNNベースのネットワークを使用し、画像からのポリシー学習が可能です。

<img src="https://github.com/kyo0221/camera_based_rl_isaac/blob/main/gif/sample_image.png" width="700">

## Example

### 🎮 1. demo
以下のコマンドで3つの並列環境を起動し、学習を行わずにシミュレーション環境のみを確認できます。

```
python create_simulator_env.py --enable_cameras --num_envs 3
```

### 🏋️ 2. training
以下のように、カスタムタスクと学習スクリプトを Isaac Lab の所定のディレクトリにコピーしてください。

🔧 タスク用コードの配置

```
cp -r ~/camera_based_rl_isaac/camera_based_rl ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic
```

🎓 トレーニング用スクリプトの配置

```
cp ~/camera_based_rl_isaac/train.py ~/IsaacLab/scripts/reinforcement_learning/sb3
```

<p align="center">
  <img src="https://github.com/kyo0221/camera_based_rl_isaac/blob/main/gif/task_path.gif" width="500">
  <img src="https://github.com/kyo0221/camera_based_rl_isaac/blob/main/gif/train_path.gif" width="500">
</p>



学習スクリプトを実行し、カメラ画像を観測すると共に並列環境で訓練を開始します
```
cd ~/IsaacLab/scripts/reinforcement_learning/sb3
python train.py --task camera_based_rl --enable_cameras --num_envs 3
```

## LICENSE
このプロジェクトのソースコードは [BSD3-Clause License](https://github.com/kyo0221/camera_based_rl_isaac/blob/feat/train/LICENSE) に基づいて公開されています

また, 一部に Isaac Lab（[BSD 3-Clause License](https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE)）のコードを含んでいます。  
詳細は LICENSE ファイルをご確認ください。
