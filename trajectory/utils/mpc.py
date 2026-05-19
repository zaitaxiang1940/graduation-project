import torch
import torch.nn as nn

class MPCWrapper:
    def __init__(self, max_acc=3.0, min_acc=-4.5, max_steer=0.5):
        self.max_acc = max_acc
        self.min_acc = min_acc
        self.max_steer = max_steer

    def project_action(self, action):
        """
        将动作投影到物理安全边界内
        action: [batch_size, 2] (acc, steer)
        """
        acc = torch.clamp(action[:, 0], self.min_acc, self.max_acc)
        steer = torch.clamp(action[:, 1], -self.max_steer, self.max_steer)
        return torch.stack([acc, steer], dim=-1)

    def enforce_safety_envelope(self, action, state, ttc):
        """
        结合TTC进行防碰撞包络截断
        """
        # 简化版逻辑：若TTC极小，强制最大减速
        danger_mask = (ttc < 1.5)
        safe_action = self.project_action(action)
        safe_action[danger_mask, 0] = self.min_acc
        return safe_action
