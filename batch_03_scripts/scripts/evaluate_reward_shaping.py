import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from _settings import Settings


@dataclass
class TrialResult:
    reward_function: str
    seed: int
    episode_rewards: List[float]
    crash_rate: float
    merge_rate: float

    def summary(self) -> Dict:
        r = np.asarray(self.episode_rewards, dtype=np.float64)
        return {
            "reward_function": self.reward_function,
            "seed": int(self.seed),
            "episodes": int(len(self.episode_rewards)),
            "mean": float(np.mean(r)),
            "std": float(np.std(r)),
            "min": float(np.min(r)),
            "p10": float(np.percentile(r, 10)),
            "p50": float(np.percentile(r, 50)),
            "p90": float(np.percentile(r, 90)),
            "max": float(np.max(r)),
            "crash_rate": float(self.crash_rate),
            "merge_rate": float(self.merge_rate),
        }


def _set_global_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _compute_episode_total_reward(episode_stats, reward_function) -> float:
    state_history = episode_stats["state_history"]
    jerk_history = episode_stats["jerk_history"]
    crashed = bool(episode_stats["crashed"])
    merged = bool(episode_stats["merged"])

    if len(state_history) == 0:
        return 0.0

    if hasattr(reward_function, "reset"):
        reward_function.reset(state_history[0])

    total = 0.0
    for i in range(len(state_history) - 1):
        next_state = state_history[i + 1]
        next_jerk = jerk_history[i + 1]
        is_last = (i == len(state_history) - 2)
        total += float(
            reward_function(
                next_state,
                next_jerk,
                crashed if is_last else False,
                merged if is_last else False,
            )
        )
    return total


def _run_trial(reward_function_name: str, seed: int, num_episodes: int, max_episode_length_s: float) -> TrialResult:
    import sumo
    import control
    import prediction
    import dqn

    Settings.SEED = seed
    Settings.REWARD_FUNCTION = reward_function_name
    _set_global_seeds(seed)

    sumo.start_sumo()
    try:
        reward_function = dqn.get_reward_function()
        episode_rewards: List[float] = []
        crashes = 0
        merges = 0

        def control_policy(state: prediction.HighwayState):
            target_speed = min(float(Settings.DESIRED_SPEED), 12.0)
            ego_x, _ = state.ego_position
            car_ahead, _ = state.get_closest_cars()
            if car_ahead is not None:
                ahead_x, ahead_speed, _ = car_ahead
                front_distance = float(ahead_x - ego_x - Settings.CAR_LENGTH)
                if front_distance < 12.0:
                    target_speed = min(target_speed, max(0.0, float(ahead_speed) - 1.0))
                elif front_distance < 20.0:
                    target_speed = min(target_speed, float(ahead_speed) + 0.5)
            control.set_ego_speed(float(np.clip(target_speed, 0.0, float(Settings.MAX_SPEED))))
            return target_speed

        for ep in range(num_episodes):
            episode_stats = control.run_episode(
                control_function=control_policy,
                state_function=prediction.HighwayState.from_sumo,
                max_episode_length=max_episode_length_s,
                wait_before_start=20,
                limit_metrics=True,
            )
            episode_rewards.append(_compute_episode_total_reward(episode_stats, reward_function))
            crashes += int(bool(episode_stats["crashed"]))
            merges += int(bool(episode_stats["merged"]))
            if (ep + 1) % 50 == 0:
                r = np.asarray(episode_rewards, dtype=np.float64)
                print(
                    json.dumps(
                        {
                            "reward_function": reward_function_name,
                            "seed": seed,
                            "episodes_done": int(ep + 1),
                            "mean_total_reward_so_far": float(np.mean(r)),
                            "crash_rate_so_far": float(crashes / float(ep + 1)),
                            "merge_rate_so_far": float(merges / float(ep + 1)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        return TrialResult(
            reward_function=reward_function_name,
            seed=seed,
            episode_rewards=episode_rewards,
            crash_rate=crashes / float(num_episodes),
            merge_rate=merges / float(num_episodes),
        )
    finally:
        sumo.exit_sumo()


def _write_csv(path: str, rewards: List[float]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i, float(r)])


def _write_results_csv(path: str, trials: List[TrialResult]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reward_function", "seed", "trial", "episode", "total_reward"])
        for trial_i, t in enumerate(trials):
            for ep_i, r in enumerate(t.episode_rewards):
                writer.writerow([t.reward_function, int(t.seed), int(trial_i), int(ep_i), float(r)])


def _plot(out_png: str, baseline: List[float], shaped: List[float], title: str, ylabel: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    w, h = 1000, 400
    m = 50

    all_y = baseline + shaped
    if not all_y:
        return
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0

    def xy(points: List[float]) -> str:
        n = max(1, len(points) - 1)
        out = []
        for i, y in enumerate(points):
            x_pix = m + (w - 2 * m) * (i / n)
            y_norm = (float(y) - y_min) / (y_max - y_min)
            y_pix = h - m - (h - 2 * m) * y_norm
            out.append(f"{x_pix:.2f},{y_pix:.2f}")
        return " ".join(out)

    baseline_poly = xy(baseline)
    shaped_poly = xy(shaped)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>
<text x="{m}" y="{m - 20}" font-family="sans-serif" font-size="16">{title}</text>
<text x="{m}" y="{h - 10}" font-family="sans-serif" font-size="12">episode</text>
<text x="10" y="{m}" font-family="sans-serif" font-size="12" transform="rotate(-90 10,{m})">{ylabel}</text>
<rect x="{m}" y="{m}" width="{w - 2*m}" height="{h - 2*m}" fill="none" stroke="#999" stroke-width="1"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="1.5" points="{baseline_poly}"/>
<polyline fill="none" stroke="#ff7f0e" stroke-width="1.5" points="{shaped_poly}"/>
<text x="{w - m - 140}" y="{m + 16}" font-family="sans-serif" font-size="12" fill="#1f77b4">baseline</text>
<text x="{w - m - 140}" y="{m + 34}" font-family="sans-serif" font-size="12" fill="#ff7f0e">shaped</text>
</svg>"""
    with open(out_png, "w") as f:
        f.write(svg)


def _plot_multi(out_svg: str, series: Dict[str, List[float]], title: str, ylabel: str):
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    w, h = 1000, 400
    m = 50
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    all_y = []
    for v in series.values():
        all_y.extend(v)
    if not all_y:
        return
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0

    def xy(points: List[float]) -> str:
        n = max(1, len(points) - 1)
        out = []
        for i, y in enumerate(points):
            x_pix = m + (w - 2 * m) * (i / n)
            y_norm = (float(y) - y_min) / (y_max - y_min)
            y_pix = h - m - (h - 2 * m) * y_norm
            out.append(f"{x_pix:.2f},{y_pix:.2f}")
        return " ".join(out)

    polylines = []
    legend = []
    for i, (name, points) in enumerate(series.items()):
        color = colors[i % len(colors)]
        polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{xy(points)}"/>')
        legend.append(
            f'<text x="{w - m - 180}" y="{m + 16 + i*18}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>
<text x="{m}" y="{m - 20}" font-family="sans-serif" font-size="16">{title}</text>
<text x="{m}" y="{h - 10}" font-family="sans-serif" font-size="12">episode</text>
<text x="10" y="{m}" font-family="sans-serif" font-size="12" transform="rotate(-90 10,{m})">{ylabel}</text>
<rect x="{m}" y="{m}" width="{w - 2*m}" height="{h - 2*m}" fill="none" stroke="#999" stroke-width="1"/>
{''.join(polylines)}
{''.join(legend)}
</svg>"""
    with open(out_svg, "w") as f:
        f.write(svg)


def _rolling_mean(x: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(x)
    arr = np.asarray(x, dtype=np.float64)
    c = np.cumsum(np.insert(arr, 0, 0.0))
    out = (c[window:] - c[:-window]) / float(window)
    pad = np.full(window - 1, out[0] if out.size else 0.0)
    return list(np.concatenate([pad, out]))



def _rolling_var(x: List[float], window: int) -> List[float]:
    if window <= 1:
        return [0.0 for _ in x]
    arr = np.asarray(x, dtype=np.float64)
    c1 = np.cumsum(np.insert(arr, 0, 0.0))
    c2 = np.cumsum(np.insert(arr * arr, 0, 0.0))
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    mean = s1 / float(window)
    var = (s2 / float(window)) - mean * mean
    var = np.maximum(var, 0.0)
    pad = np.full(window - 1, var[0] if var.size else 0.0)
    return list(np.concatenate([pad, var]))


def _plot_hist_multi(out_svg: str, series: Dict[str, List[float]], title: str):
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    w, h = 900, 400
    m = 50
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    all_vals = []
    for v in series.values():
        all_vals.extend(v)
    if not all_vals:
        return

    all_r = np.asarray(all_vals, dtype=np.float64)
    r_min = float(np.min(all_r))
    r_max = float(np.max(all_r))
    if r_max - r_min < 1e-9:
        r_max = r_min + 1.0

    bins = 60
    edges = np.linspace(r_min, r_max, bins + 1)
    hists = []
    max_count = 1.0
    for name, vals in series.items():
        hist, _ = np.histogram(np.asarray(vals, dtype=np.float64), bins=edges)
        hists.append((name, hist))
        max_count = max(max_count, float(np.max(hist)))

    bar_w = (w - 2 * m) / bins
    rects = []
    for i in range(bins):
        x0 = m + i * bar_w
        for j, (name, hist) in enumerate(hists):
            color = colors[j % len(colors)]
            hh = (h - 2 * m) * (float(hist[i]) / max_count)
            rects.append(
                f'<rect x="{x0:.2f}" y="{h - m - hh:.2f}" width="{bar_w:.2f}" height="{hh:.2f}" fill="{color}" opacity="0.25"/>'
            )

    legend = []
    for j, (name, _) in enumerate(hists):
        color = colors[j % len(colors)]
        legend.append(
            f'<text x="{w - m - 180}" y="{m + 16 + j*18}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>
<text x="{m}" y="{m - 20}" font-family="sans-serif" font-size="16">{title}</text>
<text x="{m}" y="{h - 10}" font-family="sans-serif" font-size="12">episode total reward</text>
<text x="10" y="{m}" font-family="sans-serif" font-size="12" transform="rotate(-90 10,{m})">count</text>
<rect x="{m}" y="{m}" width="{w - 2*m}" height="{h - 2*m}" fill="none" stroke="#999" stroke-width="1"/>
{''.join(rects)}
{''.join(legend)}
</svg>"""
    with open(out_svg, "w") as f:
        f.write(svg)


def _aggregate(trials: List[TrialResult]) -> Tuple[List[float], Dict]:
    rewards = []
    summaries = []
    for t in trials:
        rewards.extend(t.episode_rewards)
        summaries.append(t.summary())
    all_rewards = np.asarray(rewards, dtype=np.float64)
    aggregate = {
        "trials": summaries,
        "aggregate": {
            "episodes": int(all_rewards.size),
            "mean": float(np.mean(all_rewards)) if all_rewards.size else 0.0,
            "std": float(np.std(all_rewards)) if all_rewards.size else 0.0,
            "min": float(np.min(all_rewards)) if all_rewards.size else 0.0,
            "p10": float(np.percentile(all_rewards, 10)) if all_rewards.size else 0.0,
            "p50": float(np.percentile(all_rewards, 50)) if all_rewards.size else 0.0,
            "p90": float(np.percentile(all_rewards, 90)) if all_rewards.size else 0.0,
            "max": float(np.max(all_rewards)) if all_rewards.size else 0.0,
        },
    }
    return rewards, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max_episode_length_s", type=float, default=50.0)
    parser.add_argument("--baseline_reward", default="Slotted Jerk")
    parser.add_argument("--shaped_reward", default="Shaped")
    parser.add_argument("--stable_reward", default="Shaped Stable")
    parser.add_argument("--outdir", default="reports/reward_shaping")
    parser.add_argument("--rolling_window", type=int, default=50)
    args = parser.parse_args()

    if args.config:
        Settings.load_from_file(args.config)

    base_seeds = []
    if Settings.SEED == "Random":
        base_seed = 0
    else:
        base_seed = int(Settings.SEED)
    for i in range(int(args.trials)):
        base_seeds.append(base_seed + i)

    baseline_trials = []
    shaped_trials = []
    stable_trials = []
    for seed in base_seeds:
        baseline_trials.append(
            _run_trial(args.baseline_reward, seed, int(args.episodes), float(args.max_episode_length_s))
        )
        shaped_trials.append(
            _run_trial(args.shaped_reward, seed, int(args.episodes), float(args.max_episode_length_s))
        )
        stable_trials.append(
            _run_trial(args.stable_reward, seed, int(args.episodes), float(args.max_episode_length_s))
        )

    baseline_rewards, baseline_stats = _aggregate(baseline_trials)
    shaped_rewards, shaped_stats = _aggregate(shaped_trials)
    stable_rewards, stable_stats = _aggregate(stable_trials)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "baseline_summary.json"), "w") as f:
        json.dump(baseline_stats, f, indent=2, sort_keys=True)
    with open(os.path.join(args.outdir, "shaped_summary.json"), "w") as f:
        json.dump(shaped_stats, f, indent=2, sort_keys=True)
    with open(os.path.join(args.outdir, "stable_summary.json"), "w") as f:
        json.dump(stable_stats, f, indent=2, sort_keys=True)

    _write_csv(os.path.join(args.outdir, "baseline_rewards.csv"), baseline_rewards)
    _write_csv(os.path.join(args.outdir, "shaped_rewards.csv"), shaped_rewards)
    _write_csv(os.path.join(args.outdir, "stable_rewards.csv"), stable_rewards)
    _write_results_csv(
        os.path.join(args.outdir, "results.csv"),
        baseline_trials + shaped_trials + stable_trials,
    )

    _plot_multi(
        os.path.join(args.outdir, "reward_curve.svg"),
        {
            "baseline": baseline_rewards,
            "shaped": shaped_rewards,
            "stable": stable_rewards,
        },
        title="Episode Total Reward",
        ylabel="total reward",
    )
    _plot_multi(
        os.path.join(args.outdir, "reward_curve_ma.svg"),
        {
            "baseline": _rolling_mean(baseline_rewards, int(args.rolling_window)),
            "shaped": _rolling_mean(shaped_rewards, int(args.rolling_window)),
            "stable": _rolling_mean(stable_rewards, int(args.rolling_window)),
        },
        title=f"Episode Total Reward (rolling mean, window={int(args.rolling_window)})",
        ylabel="total reward (MA)",
    )
    _plot_multi(
        os.path.join(args.outdir, "reward_var_ma.svg"),
        {
            "baseline": _rolling_var(baseline_rewards, int(args.rolling_window)),
            "shaped": _rolling_var(shaped_rewards, int(args.rolling_window)),
            "stable": _rolling_var(stable_rewards, int(args.rolling_window)),
        },
        title=f"Rolling Variance (window={int(args.rolling_window)})",
        ylabel="variance",
    )
    _plot_hist_multi(
        os.path.join(args.outdir, "reward_hist.svg"),
        {
            "baseline": baseline_rewards,
            "shaped": shaped_rewards,
            "stable": stable_rewards,
        },
        title="Reward Distribution (Episode Total Reward)",
    )

    report = {
        "baseline": baseline_stats,
        "shaped": shaped_stats,
        "stable": stable_stats,
        "delta_mean": float(shaped_stats["aggregate"]["mean"] - baseline_stats["aggregate"]["mean"]),
        "delta_mean_stable": float(stable_stats["aggregate"]["mean"] - baseline_stats["aggregate"]["mean"]),
    }
    with open(os.path.join(args.outdir, "comparison.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
