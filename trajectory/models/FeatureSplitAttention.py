import numpy as np
import math
import time
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F



class FeatureSplitAttention(nn.Module):
    def __init__(self,
                 input_dim=23,
                 a_dims=16,  # 0-15维
                 b_dims=7,  # 16-22维
                 d_model=128,
                 n_heads=4):
        super().__init__()
        self.a_dims = a_dims
        self.b_dims = b_dims

        # 投影层定义
        self.proj_a_kv = nn.Linear(a_dims, 2 * d_model)  # A生成K,V
        self.proj_b_qkv = nn.Linear(b_dims, 3 * d_model)  # B生成Q,K,V

        # 多头注意力参数
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head * n_heads == d_model

        # 输出融合层
        self.out_proj = nn.Linear(d_model, b_dims)  # 保持维度一致

    def forward(self, x):
        """
        输入形状: (batch_size, seq_len, 23)
        输出形状: (batch_size, seq_len, 23)
        """
        # 分割特征
        A = x[..., :self.a_dims]  # (B, T, 16)
        B = x[..., self.a_dims:]  # (B, T, 7)

        # 生成注意力组件
        ## A生成K,V
        kv_A = self.proj_a_kv(A)  # (B, T, 2*d_model)
        K_A, V_A = torch.split(kv_A, [self.d_model] * 2, dim=-1)

        ## B生成Q,K,V
        qkv_B = self.proj_b_qkv(B)  # (B, T, 3*d_model)
        Q_B, K_B, V_B = torch.split(qkv_B, [self.d_model] * 3, dim=-1)

        # 合并K,V（A与B的K,V拼接）
        K = torch.cat([K_A, K_B], dim=1)  # (B, 2*T, d_model)
        V = torch.cat([V_A, V_B], dim=1)

        # 多头处理
        Q = Q_B.view(-1, Q_B.size(1), self.n_heads, self.d_head).permute(0, 2, 1, 3)  # (B, H, T, D)
        K = K.view(-1, K.size(1), self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = V.view(-1, V.size(1), self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # 注意力计算
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, T, D)

        # 恢复形状
        out = out.permute(0, 2, 1, 3).contiguous().view(*B.shape[:2], -1)  # (B, T, d_model)

        # 残差连接 + 维度还原
        out = self.out_proj(out) + B  # (B, T, 7)

        # 合并回完整特征
        return torch.cat([A, out], dim=-1)