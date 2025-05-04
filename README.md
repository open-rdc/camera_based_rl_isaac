# camera_baed_rl_isaac

## Overview
本プロジェクトは、Isaac Lab を基盤とした強化学習環境で、RGB画像を観測として使用し、並列に複数エージェントを学習させるためのシステムです。
カスタムタスク camera_based_rl において、画像入力から特徴ベクトルを抽出するCNNベースのネットワークを使用し、画像からのポリシー学習が可能です。

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
cp -p ~/camera_based_rl_isaac/camera_based_rl ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic
```

🎓 トレーニング用スクリプトの配置

```
cp ~/camera_based_rl_isaac/train.py ~/IsaacLab/scripts/reinforcement_learning/sb3
```

demo movie
![demo](https://github.com/kyo0221/camera_based_rl_isaac/blob/feat/train/gif/task_path.gif)
![demo](https://github.com/kyo0221/camera_based_rl_isaac/blob/feat/train/gif/train_path.gif)


学習スクリプトを実行し、カメラ画像を観測すると共に並列環境で訓練を開始します
```
cd ~/IsaacLab/scripts/reinforcement_learning/sb3
python train.py --task camera_based_rl --enable_cameras --num_envs 3
```
