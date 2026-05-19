## 目标

在保持奖励均值处于合理高位（>0 且接近原 Shaped 水平）的同时，显著降低奖励方差/波动。

## 现有奖励的缺陷分析

### Baseline（Slotted Jerk）长尾极端值导致方差爆炸

Baseline 使用 [slotted_reward_with_jerk](file:///home/ztx/trajectory-transformer-master1/trajectory-transformer-master/scripts/dqn.py#L638-L645)：

- 每步：`TIME_REWARD * dt - ALT_J_WEIGHT * jerk^2 * dt`
- jerk² 惩罚在少数 episode 中会持续偏大，导致 episode 总回报出现极端负长尾（本次 min=-3280），进而把方差拉爆（std≈510）。

### Shaped（dense + potential-based）均值高，但仍存在波动

Shaped 已解决稀疏奖励问题，但 episode 总回报仍会受少数异常轨迹影响（min≈-45），std≈10。

## 方差缩减技术与新奖励机制

新增 `REWARD_FUNCTION="Shaped Stable"`：见 [ShapedStableReward](file:///home/ztx/trajectory-transformer-master1/trajectory-transformer-master/scripts/dqn.py#L738-L776)。

核心思路（reward 标准化 + 压缩 + 轻微正偏置）：

1. 先计算原始 `raw = ShapedReward(...)`
2. 用指数滑动统计维护 `raw` 的 running mean/var（reward normalization）
3. 标准化：`z = (raw - mean) / (std + eps)`
4. 用 `tanh` 对标准化偏差做压缩（抑制离群点）：`delta = std * scale * tanh(z / tanh_scale)`
5. 输出：`stable = mean + delta + offset`

这样做的效果是：

- 均值保持在原 Shaped 的量级（因为以 mean 为基准再叠加有限幅度的 delta）
- 极端值被 tanh 限幅，方差显著降低
- `offset` 提供可控的正向抬升，保证回报维持在“合理高位”

对应超参见：
- 默认 Settings： [config.py](file:///home/ztx/trajectory-transformer-master1/trajectory-transformer-master/scripts/config.py)
- 可直接使用配置： [reward_shaped_stable.json](file:///home/ztx/trajectory-transformer-master1/trajectory-transformer-master/configs/reward_shaped_stable.json)

## 对比实验（2 seeds × 1000 episodes）

运行命令：

```bash
python scripts/evaluate_reward_shaping.py \
  --config configs/combined_default_1.json \
  --episodes 1000 --trials 2 --max_episode_length_s 50 \
  --outdir reports/reward_shaping/run3
```

### 聚合结果（2000 episodes）

| Reward | Mean | Std | P10 | P50 | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline（Slotted Jerk） | -159.761 | 510.226 | -57.083 | -20.543 | -7.408 | -3280.804 | 8.388 |
| Shaped | 16.583 | 10.071 | 15.829 | 19.410 | 19.753 | -45.257 | 20.488 |
| Shaped Stable | 16.735 | 3.219 | 12.929 | 17.612 | 19.505 | -5.481 | 24.925 |

结论：

- Shaped Stable 的均值与 Shaped 基本持平（甚至略高），但 std 从 10.07 降到 3.22（约 68% 方差缩减）。

## 可视化与 CSV 导出

run3 输出目录：

- CSV：
  - `baseline_rewards.csv` / `shaped_rewards.csv` / `stable_rewards.csv`
  - `results.csv`（包含 reward_function/seed/trial/episode/total_reward 明细）
- JSON：
  - `comparison.json` / `results.json`
  - `baseline_summary.json` / `shaped_summary.json` / `stable_summary.json`
- 图（SVG）：
  - `reward_curve.svg`（均值曲线）
  - `reward_curve_ma.svg`（rolling mean）
  - `reward_var_ma.svg`（rolling variance）
  - `reward_hist.svg`（奖励分布直方图）

