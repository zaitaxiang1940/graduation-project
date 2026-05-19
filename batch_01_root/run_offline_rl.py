import os
import sys
import torch
import numpy as np
import argparse

# 添加项目路径到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 简单的环境模拟类
class SimpleEnv:
    def __init__(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_episode_steps = 100
        self.current_step = 0
    
    def reset(self):
        self.current_step = 0
        return np.random.rand(self.observation_dim)
    
    def step(self, action):
        self.current_step += 1
        next_observation = np.random.rand(self.observation_dim)
        reward = np.random.rand()
        terminal = self.current_step >= self.max_episode_steps
        return next_observation, reward, terminal, {}

# 策略网络类
class PolicyNetwork(torch.nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(observation_dim, 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.mean_linear = torch.nn.Linear(256, action_dim)
        self.log_std_linear = torch.nn.Linear(256, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std
    
    def act(self, observation, deterministic=True):
        with torch.no_grad():
            x = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            mean, std = self.forward(x)
            if deterministic:
                action = mean
            else:
                action = mean + std * torch.randn_like(std)
            action = torch.tanh(action)
        return action.squeeze(0).numpy()

# 加载模型并运行离线强化学习
def run_offline_rl(model_path, n_episodes=5):
    # 加载模型
    try:
        data = torch.load(model_path)
        print(f"Successfully loaded {model_path}")
        print(f"Model contains: {list(data.keys())}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 从状态字典中获取网络维度
    policy_state_dict = data['policy_state_dict']
    observation_dim = policy_state_dict['linear1.weight'].shape[1]
    action_dim = policy_state_dict['mean_linear.weight'].shape[0]
    
    print(f"\nNetwork dimensions:")
    print(f"Observation dimension: {observation_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Hidden layer size: {policy_state_dict['linear1.weight'].shape[0]}")
    
    # 创建策略网络
    policy = PolicyNetwork(observation_dim, action_dim)
    
    # 加载策略网络的状态字典
    if 'policy_state_dict' in data:
        policy.load_state_dict(data['policy_state_dict'])
        print("Loaded policy state dict")
    else:
        print("Warning: policy_state_dict not found in model")
        return
    
    # 创建简单环境
    env = SimpleEnv(observation_dim, action_dim)
    
    # 主循环
    for episode in range(n_episodes):
        print(f"\n=== Episode {episode + 1} ===")
        observation = env.reset()
        total_reward = 0
        
        for t in range(env.max_episode_steps):
            # 使用策略网络生成动作
            action = policy.act(observation, deterministic=True)
            
            # 执行动作
            next_observation, reward, terminal, _ = env.step(action)
            
            # 更新总奖励
            total_reward += reward
            
            # 打印信息
            print(f"Step {t}: Action = {action}..., Reward = {reward:.4f}, Total Reward = {total_reward:.4f}")
            
            if terminal:
                break
            
            observation = next_observation
        
        print(f"Episode {episode + 1} finished with total reward: {total_reward:.4f}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run offline RL with policy model")
    parser.add_argument("model_path", type=str, help="Path to the .pkl model file")
    parser.add_argument("--n_episodes", type=int, default=5, help="Number of episodes to run")
    
    args = parser.parse_args()
    
    # 运行离线强化学习
    run_offline_rl(
        args.model_path,
        args.n_episodes
    )
