# isaaclab_tasks/my_custom_task/task.py

from camera_based_rl_isaac.tasks.reward import compute_reward

class MyCustomTask:
    def step(self, action):
        # 環境のステップ処理
        obs, reward, done, info = self.env.step(action)

        # 報酬の再計算
        reward = compute_reward(obs, action, next_obs, info)

        return obs, reward, done, info
