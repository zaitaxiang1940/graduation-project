## 结论（两轮测试：seed=3/4）

对比 2 轮 × 1000 episode（共 2000 episode）：

| Reward | 平均回报 | P50 | P10 | P90 | 最小值 | 最大值 | Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline（Slotted Jerk） | -159.761 | -20.543 | -57.083 | -7.408 | -3280.804 | 8.388 | 510.226 |
| Shaped（Shaped） | 16.583 | 19.410 | 15.829 | 19.753 | -45.257 | 20.488 | 10.071 |

Shaped 的 episode 平均回报稳定大于 0，并且方差大幅降低。

## 产物

- 对比数据：[comparison.json](file:///home/ztx/trajectory-transformer-master1/trajectory-transformer-master/reports/reward_shaping/run2/comparison.json)
- 曲线/分布（SVG）：`reward_curve.svg`、`reward_curve_ma.svg`、`reward_hist.svg`
- 明细 CSV：`baseline_rewards.csv`、`shaped_rewards.csv`

