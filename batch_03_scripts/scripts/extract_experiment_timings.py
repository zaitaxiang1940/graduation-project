import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _parse_iso8601(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _dt_to_iso(dt: datetime) -> str:
    return _to_utc(dt).isoformat()


def _mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


@dataclass
class RunTiming:
    run_dir: Path
    results_json: Path
    results_csv: Optional[Path]
    steps_csv: Optional[Path]

    dataset: str
    exp_name: str
    suffix: str
    gpt_epoch: str

    start_ts_utc: datetime
    end_ts_utc: datetime
    elapsed_wall_s: float

    cpu_reward_s: float
    cpu_state_s: float
    cpu_vector_s: float
    cpu_planning_s: float
    cpu_total_known_s: float

    episodes: int
    total_steps: int

    validation_errors: List[str]

    def condition_key(self) -> str:
        return f"{self.dataset}::{self.exp_name}"

    def to_row(self) -> Dict[str, Any]:
        return {
            "condition_key": self.condition_key(),
            "dataset": self.dataset,
            "exp_name": self.exp_name,
            "suffix": self.suffix,
            "gpt_epoch": self.gpt_epoch,
            "run_dir": str(self.run_dir),
            "start_ts_utc": _dt_to_iso(self.start_ts_utc),
            "end_ts_utc": _dt_to_iso(self.end_ts_utc),
            "elapsed_wall_s": float(self.elapsed_wall_s),
            "cpu_reward_s": float(self.cpu_reward_s),
            "cpu_state_s": float(self.cpu_state_s),
            "cpu_vector_s": float(self.cpu_vector_s),
            "cpu_planning_s": float(self.cpu_planning_s),
            "cpu_total_known_s": float(self.cpu_total_known_s),
            "episodes": int(self.episodes),
            "total_steps": int(self.total_steps),
            "validation_errors": ";".join(self.validation_errors),
        }


def _find_run_artifacts(run_dir: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    results_json = run_dir / "results.json"
    results_csv = run_dir / "results.csv"
    steps_csv = run_dir / "steps.csv"
    return (
        results_json,
        results_csv if results_csv.exists() else None,
        steps_csv if steps_csv.exists() else None,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _infer_start_end_ts(meta: Dict[str, Any], artifacts: Iterable[Path]) -> Tuple[datetime, datetime]:
    start = None
    ts = meta.get("timestamp_utc")
    if isinstance(ts, str):
        start = _parse_iso8601(ts)
    if start is None:
        start = min(_mtime_utc(p) for p in artifacts)
    end = max(_mtime_utc(p) for p in artifacts)
    return _to_utc(start), _to_utc(end)


def _sum_cpu_times(episodes: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    reward_s = 0.0
    state_s = 0.0
    vector_s = 0.0
    planning_s = 0.0
    for ep in episodes:
        reward_s += _safe_float(ep.get("reward_time_total_s", 0.0))
        state_s += _safe_float(ep.get("state_time_total_s", 0.0))
        vector_s += _safe_float(ep.get("vector_time_total_s", 0.0))
        planning_s += _safe_float(ep.get("planning_time_avg_s", 0.0)) * _safe_float(ep.get("planning_calls", 0.0))
    return reward_s, state_s, vector_s, planning_s


def _validate_elapsed(elapsed_wall_s: float, max_duration_s: float) -> List[str]:
    errors = []
    if elapsed_wall_s < 0:
        errors.append("negative_elapsed")
    if elapsed_wall_s > max_duration_s:
        errors.append("elapsed_too_large")
    if elapsed_wall_s == 0:
        errors.append("elapsed_zero")
    return errors


def discover_runs(root: Path) -> List[Path]:
    return sorted({p.parent for p in root.rglob("results.json")})


def parse_run(run_dir: Path, max_duration_s: float) -> Optional[RunTiming]:
    results_json, results_csv, steps_csv = _find_run_artifacts(run_dir)
    if not results_json.exists():
        return None

    data = _load_json(results_json)
    meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}
    args = data.get("args", {}) if isinstance(data.get("args", {}), dict) else {}
    episodes = data.get("episodes", [])
    if not isinstance(episodes, list):
        episodes = []

    artifacts = [results_json]
    if results_csv is not None:
        artifacts.append(results_csv)
    if steps_csv is not None:
        artifacts.append(steps_csv)

    start_ts, end_ts = _infer_start_end_ts(meta, artifacts)
    elapsed_wall_s = (end_ts - start_ts).total_seconds()

    cpu_reward_s, cpu_state_s, cpu_vector_s, cpu_planning_s = _sum_cpu_times(episodes)
    cpu_total_known_s = cpu_reward_s + cpu_state_s + cpu_vector_s + cpu_planning_s

    total_steps = sum(_safe_int(ep.get("steps", 0)) for ep in episodes)
    validation_errors = _validate_elapsed(elapsed_wall_s, max_duration_s)

    return RunTiming(
        run_dir=run_dir,
        results_json=results_json,
        results_csv=results_csv,
        steps_csv=steps_csv,
        dataset=str(args.get("dataset", "")),
        exp_name=str(args.get("exp_name", "")),
        suffix=str(args.get("suffix", "")),
        gpt_epoch=str(meta.get("gpt_epoch", "")),
        start_ts_utc=start_ts,
        end_ts_utc=end_ts,
        elapsed_wall_s=float(elapsed_wall_s),
        cpu_reward_s=float(cpu_reward_s),
        cpu_state_s=float(cpu_state_s),
        cpu_vector_s=float(cpu_vector_s),
        cpu_planning_s=float(cpu_planning_s),
        cpu_total_known_s=float(cpu_total_known_s),
        episodes=int(len(episodes)),
        total_steps=int(total_steps),
        validation_errors=validation_errors,
    )


def extract_episode_rows(run: RunTiming) -> List[Dict[str, Any]]:
    data = _load_json(run.results_json)
    episodes = data.get("episodes", [])
    if not isinstance(episodes, list):
        return []
    out = []
    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        planning_calls = _safe_int(ep.get("planning_calls", 0))
        planning_time_avg_s = _safe_float(ep.get("planning_time_avg_s", 0.0))
        out.append(
            {
                "condition_key": run.condition_key(),
                "dataset": run.dataset,
                "exp_name": run.exp_name,
                "suffix": run.suffix,
                "gpt_epoch": run.gpt_epoch,
                "run_dir": str(run.run_dir),
                "start_ts_utc": _dt_to_iso(run.start_ts_utc),
                "end_ts_utc": _dt_to_iso(run.end_ts_utc),
                "elapsed_wall_s": float(run.elapsed_wall_s),
                "episode": _safe_int(ep.get("episode", 0)),
                "steps": _safe_int(ep.get("steps", 0)),
                "total_reward": _safe_float(ep.get("total_reward", 0.0)),
                "avg_control": _safe_float(ep.get("avg_control", 0.0)),
                "avg_speed": _safe_float(ep.get("avg_speed", 0.0)),
                "avg_jerk": _safe_float(ep.get("avg_jerk", 0.0)),
                "reward_time_total_s": _safe_float(ep.get("reward_time_total_s", 0.0)),
                "state_time_total_s": _safe_float(ep.get("state_time_total_s", 0.0)),
                "vector_time_total_s": _safe_float(ep.get("vector_time_total_s", 0.0)),
                "planning_calls": planning_calls,
                "planning_time_avg_s": planning_time_avg_s,
                "planning_time_total_s": float(planning_calls) * planning_time_avg_s,
            }
        )
    return out


def extract_step_rows(run: RunTiming, max_rows: int = 0) -> List[Dict[str, Any]]:
    if run.steps_csv is None:
        return []
    rows = []
    with open(run.steps_csv, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(
                {
                    "condition_key": run.condition_key(),
                    "dataset": run.dataset,
                    "exp_name": run.exp_name,
                    "suffix": run.suffix,
                    "gpt_epoch": run.gpt_epoch,
                    "run_dir": str(run.run_dir),
                    "start_ts_utc": _dt_to_iso(run.start_ts_utc),
                    "episode": _safe_int(row.get("episode", 0)),
                    "step": _safe_int(row.get("step", 0)),
                    "action": row.get("action", ""),
                    "reward": _safe_float(row.get("reward", 0.0)),
                    "total_reward": _safe_float(row.get("total_reward", 0.0)),
                    "planning_time_s": _safe_float(row.get("planning_time_s", 0.0)),
                    "terminal": row.get("terminal", ""),
                }
            )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["empty"])
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _plot_multi(out_svg: Path, series: Dict[str, List[float]], title: str, ylabel: str):
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1000, 400
    m = 50
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

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
            f'<text x="{w - m - 220}" y="{m + 16 + i*18}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>
<text x="{m}" y="{m - 20}" font-family="sans-serif" font-size="16">{title}</text>
<text x="{m}" y="{h - 10}" font-family="sans-serif" font-size="12">run index (sorted by start time)</text>
<text x="10" y="{m}" font-family="sans-serif" font-size="12" transform="rotate(-90 10,{m})">{ylabel}</text>
<rect x="{m}" y="{m}" width="{w - 2*m}" height="{h - 2*m}" fill="none" stroke="#999" stroke-width="1"/>
{''.join(polylines)}
{''.join(legend)}
</svg>"""
    out_svg.write_text(svg)


def _plot_hist(out_svg: Path, values: List[float], title: str, xlabel: str):
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    w, h = 900, 400
    m = 50
    if not values:
        return
    arr = np.asarray(values, dtype=np.float64)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    if v_max - v_min < 1e-9:
        v_max = v_min + 1.0

    bins = 60
    edges = np.linspace(v_min, v_max, bins + 1)
    hist, _ = np.histogram(arr, bins=edges)
    max_count = float(max(np.max(hist), 1.0))

    bar_w = (w - 2 * m) / bins
    rects = []
    for i in range(bins):
        x0 = m + i * bar_w
        hh = (h - 2 * m) * (float(hist[i]) / max_count)
        rects.append(
            f'<rect x="{x0:.2f}" y="{h - m - hh:.2f}" width="{bar_w:.2f}" height="{hh:.2f}" fill="#1f77b4" opacity="0.45"/>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>
<text x="{m}" y="{m - 20}" font-family="sans-serif" font-size="16">{title}</text>
<text x="{m}" y="{h - 10}" font-family="sans-serif" font-size="12">{xlabel}</text>
<text x="10" y="{m}" font-family="sans-serif" font-size="12" transform="rotate(-90 10,{m})">count</text>
<rect x="{m}" y="{m}" width="{w - 2*m}" height="{h - 2*m}" fill="none" stroke="#999" stroke-width="1"/>
{''.join(rects)}
</svg>"""
    out_svg.write_text(svg)


def _summarize(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs", help="实验输出根目录（默认 logs）")
    parser.add_argument("--outdir", default="reports/condition_timings", help="导出目录")
    parser.add_argument("--max_duration_s", type=float, default=6 * 3600, help="耗时上限（用于异常检测）")
    parser.add_argument("--rolling_window", type=int, default=20, help="rolling 统计窗口（用于曲线）")
    parser.add_argument("--include_steps", action="store_true", help="合并导出逐 step 数据（可能很大）")
    parser.add_argument("--max_step_rows_per_run", type=int, default=0, help="逐 step 最大导出行数（0 表示不限制）")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(root)
    runs: List[RunTiming] = []
    for rd in run_dirs:
        rt = parse_run(rd, max_duration_s=float(args.max_duration_s))
        if rt is not None:
            runs.append(rt)

    runs.sort(key=lambda r: r.start_ts_utc)

    run_rows = [r.to_row() for r in runs]
    _write_csv(outdir / "condition_runs.csv", run_rows)

    episode_rows: List[Dict[str, Any]] = []
    for r in runs:
        episode_rows.extend(extract_episode_rows(r))
    _write_csv(outdir / "condition_episodes.csv", episode_rows)

    if args.include_steps:
        step_rows: List[Dict[str, Any]] = []
        for r in runs:
            step_rows.extend(extract_step_rows(r, max_rows=int(args.max_step_rows_per_run)))
        _write_csv(outdir / "condition_steps.csv", step_rows)

    wall = [r.elapsed_wall_s for r in runs]
    cpu = [r.cpu_total_known_s for r in runs]
    _plot_hist(outdir / "elapsed_wall_hist.svg", wall, "Wall-Clock Elapsed Time Distribution", "elapsed_wall_s")
    _plot_hist(outdir / "cpu_total_known_hist.svg", cpu, "Known CPU Time Distribution", "cpu_total_known_s")
    _plot_multi(
        outdir / "elapsed_wall_curve.svg",
        {
            "elapsed_wall_s": wall,
            f"elapsed_wall_s_ma{int(args.rolling_window)}": _rolling_mean(wall, int(args.rolling_window)),
        },
        "Wall-Clock Elapsed Time (sorted by start time)",
        "seconds",
    )
    _plot_multi(
        outdir / "elapsed_wall_var.svg",
        {
            f"var_ma{int(args.rolling_window)}": _rolling_var(wall, int(args.rolling_window)),
        },
        f"Wall-Clock Elapsed Variance (window={int(args.rolling_window)})",
        "variance",
    )

    by_condition: Dict[str, List[RunTiming]] = {}
    for r in runs:
        by_condition.setdefault(r.condition_key(), []).append(r)

    condition_rows = []
    for key, rs in sorted(by_condition.items(), key=lambda kv: kv[0]):
        vals = [x.elapsed_wall_s for x in rs]
        summary = _summarize(vals)
        planning_vals = [x.cpu_planning_s for x in rs]
        planning_summary = _summarize(planning_vals)
        condition_rows.append(
            {
                "condition_key": key,
                "runs": int(summary.get("count", 0)),
                "mean_elapsed_wall_s": float(summary.get("mean", 0.0)),
                "std_elapsed_wall_s": float(summary.get("std", 0.0)),
                "min_elapsed_wall_s": float(summary.get("min", 0.0)),
                "p50_elapsed_wall_s": float(summary.get("p50", 0.0)),
                "max_elapsed_wall_s": float(summary.get("max", 0.0)),
                "mean_cpu_planning_s": float(planning_summary.get("mean", 0.0)),
                "std_cpu_planning_s": float(planning_summary.get("std", 0.0)),
            }
        )
    _write_csv(outdir / "condition_summary.csv", condition_rows)

    errors = [r for r in runs if r.validation_errors]
    _write_csv(outdir / "validation_errors.csv", [r.to_row() for r in errors])

    report = {
        "root": str(root),
        "outdir": str(outdir),
        "runs_total": int(len(runs)),
        "conditions_total": int(len(by_condition)),
        "elapsed_wall_s": _summarize(wall),
        "cpu_total_known_s": _summarize(cpu),
        "validation_errors_total": int(len(errors)),
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    md = [
        "# 实验工况耗时导出报告",
        "",
        f"- 工况数量（unique condition_key）：{report['conditions_total']}",
        f"- 运行实例数量（results.json）：{report['runs_total']}",
        f"- 异常条目数量：{report['validation_errors_total']}",
        "",
        "## 总体耗时统计（wall-clock）",
        "",
        f"- mean={report['elapsed_wall_s'].get('mean', 0):.3f}s",
        f"- std={report['elapsed_wall_s'].get('std', 0):.3f}s",
        f"- p50={report['elapsed_wall_s'].get('p50', 0):.3f}s",
        f"- min={report['elapsed_wall_s'].get('min', 0):.3f}s",
        f"- max={report['elapsed_wall_s'].get('max', 0):.3f}s",
        "",
        "## 输出文件",
        "",
        "- condition_runs.csv：每个运行实例（含开始/结束时间戳与耗时）",
        "- condition_summary.csv：按工况聚合统计",
        "- validation_errors.csv：耗时异常/可疑条目",
        "- report.json：汇总统计（机器可读）",
        "- elapsed_wall_hist.svg：耗时分布直方图",
        "- elapsed_wall_curve.svg：耗时曲线（按开始时间排序）",
        "- elapsed_wall_var.svg：rolling 方差曲线",
        "- cpu_total_known_hist.svg：已知 CPU 计时分布（reward/state/vector/planning 合计）",
        "",
    ]
    (outdir / "report.md").write_text("\n".join(md))

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
