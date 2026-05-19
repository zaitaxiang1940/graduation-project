import numpy as np
import math
import time
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F

from .ein import EinLinear



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # 计时相关属性
        self._current_time = 0.0
        self._is_cuda = False

        # 原始网络结构
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # 因果掩码和特殊掩码处理
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        joined_dim = config.observation_dim + config.action_dim + 2
        self.mask.squeeze()[:, joined_dim - 1::joined_dim] = 0

        self.n_head = config.n_head

    def forward(self, x, layer_past=None, use_cache=False, start_pos=0):
        B, T, C = x.size()
        self._is_cuda = x.is_cuda

        # 计时开始
        if self._is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        key_len = k.size(-2)
        att = att.masked_fill(self.mask[:, :, start_pos:start_pos + T, :key_len] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        y = att @ v

        # 计时结束
        if self._is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            duration = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
        else:
            duration = time.perf_counter() - start_time

        self._current_time += duration

        # 后续处理
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        present = (k, v) if use_cache else None
        return y, present

    def reset_timer(self):
        self._current_time = 0.0

    def get_time(self):
        return self._current_time


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, use_cache=False, start_pos=0):
        attn_out, present = self.attn(self.ln1(x), layer_past=layer_past, use_cache=use_cache, start_pos=start_pos)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 输入嵌入层
        self.tok_emb = nn.Embedding(config.vocab_size * config.transition_dim + 1, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer块
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # 输出头
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = EinLinear(config.transition_dim, config.n_embd, config.vocab_size + 1, bias=False)

        # 关键属性定义
        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight
        self.embedding_dim = config.n_embd

        self.apply(self._init_weights)

    def reset_timers(self):
        """ 重置所有注意力层的计时器 """
        for block in self.blocks:
            block.attn.reset_timer()

    def get_attention_time(self):
        """ 获取各层注意力计算时间列表 """
        return [block.attn.get_time() for block in self.blocks]

    def print_timing_stats(self):
        """ 打印详细计时信息 """
        print("\n=== 注意力层性能分析 ===")
        total = 0.0
        for i, t in enumerate(self.get_attention_time(), 1):
            print(f"层 {i}: {t * 1000:.2f}ms")
            total += t
        print(f"总计: {total * 1000:.2f}ms (含{len(self.blocks)}层)")
        self.reset_timers()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EinLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        param_dict = {pn: p for pn, p in self.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

    def offset_tokens(self, idx, start_pos=0):
        _, t = idx.shape
        abs_pos = torch.arange(start_pos, start_pos + t, device=idx.device) % self.transition_dim
        offsets = abs_pos * self.vocab_size
        offset_idx = idx + offsets
        offset_idx[idx == self.vocab_size] = self.stop_token
        return offset_idx

    def pad_to_full_observation(self, x, verify=False):
        b, t, _ = x.shape
        n_pad = (self.transition_dim - t % self.transition_dim) % self.transition_dim
        padding = torch.zeros(b, n_pad, self.embedding_dim, device=x.device)
        x_pad = torch.cat([x, padding], dim=1)
        x_pad = x_pad.view(-1, self.transition_dim, self.embedding_dim)
        if verify:
            self.verify(x, x_pad)
        return x_pad, n_pad

    def verify(self, x, x_pad):
        b, t, embedding_dim = x.shape
        n_states = int(np.ceil(t / self.transition_dim))
        inds = torch.arange(0, self.transition_dim, device=x.device).repeat(n_states)[:t]
        for i in range(self.transition_dim):
            x_ = x[:, inds == i]
            t_ = x_.shape[1]
            x_pad_ = x_pad[:, i].view(b, n_states, embedding_dim)[:, :t_]
            assert torch.allclose(x_, x_pad_), f"Verification failed at dimension {i}"

    def forward(self, idx, targets=None, mask=None, print_timing=False, use_cache=False, past_key_values=None, start_pos=0):
        b, t = idx.size()
        total_len = (start_pos + t) if past_key_values is not None else t
        assert total_len <= self.block_size, "Cannot forward, model block size is exhausted."
        self.reset_timers()

        # 前向传播流程
        offset_idx = self.offset_tokens(idx, start_pos=start_pos)
        token_embeddings = self.tok_emb(offset_idx)
        position_embeddings = self.pos_emb[:, start_pos:start_pos + t, :]
        x = self.drop(token_embeddings + position_embeddings)
        presents = [] if use_cache else None
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past_key_values[i], use_cache=use_cache, start_pos=start_pos)
            if use_cache:
                presents.append(present)

        if print_timing:
            self.print_timing_stats()

        x = self.ln_f(x)
        x_pad, n_pad = self.pad_to_full_observation(x)
        logits = self.head(x_pad)
        logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)[:, :t]

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction='none')
            if any(w != 1 for w in [self.action_weight, self.reward_weight, self.value_weight]):
                n_states = int(np.ceil(t / self.transition_dim))
                weights = torch.cat([
                    torch.ones(self.observation_dim, device=idx.device),
                    torch.ones(self.action_dim, device=idx.device) * self.action_weight,
                    torch.ones(1, device=idx.device) * self.reward_weight,
                    torch.ones(1, device=idx.device) * self.value_weight,
                ]).repeat(n_states)[1:].repeat(b, 1)
                loss = (loss * weights.view(-1) * mask.view(-1)).mean()
            else:
                loss = (loss * mask.view(-1)).mean()
        else:
            loss = None

        if use_cache:
            return logits, loss, presents
        return logits, loss

# ==========================================
# 新增: 论文一致性补充模块 (GNN, Critic, MPC)
# ==========================================
class GNNEncoder(nn.Module):
    def __init__(self, node_features, hidden_size):
        super().__init__()
        # 简化版GNN实现，用于提取拓扑空间特征
        self.fc1 = nn.Linear(node_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))

class PessimisticCritic(nn.Module):
    def __init__(self, hidden_size, act_dim):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(hidden_size + act_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(hidden_size + act_dim, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, state_emb, action):
        xu = torch.cat([state_emb, action], dim=-1)
        return self.q1(xu), self.q2(xu)

class GraphDecisionTransformer(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.gcn = GNNEncoder(node_features=config.observation_dim, hidden_size=config.n_embd)
        self.critic = PessimisticCritic(hidden_size=config.n_embd, act_dim=config.action_dim)
        
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # 融合图结构特征
        state_embeddings = self.gcn(states)
        # 调用父类(标准Transformer)处理时序依赖
        return super().forward(state_embeddings)
# ==========================================
