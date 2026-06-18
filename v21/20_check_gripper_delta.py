#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 LeRobot v2.1 数据集中夹爪相邻帧 delta。

脚本只读取数据集 parquet，不修改原始数据。默认检查 0 基下标 6 和 13：
left_gripper_pos / right_gripper_pos，并同时统计 observation.state 和 action。

示例：
    python 20_check_gripper_delta.py \
        --repo_id piperx/piperx_grab_bigbox_0526_0609_nonidle \
        --root /home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_0526_0609_nonidle \
        --output_dir ./output \
        --max_episodes 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lerobot-dataset-tools")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


SIGNALS = ("observation.state", "action")
DEFAULT_DIMS = (6, 13)
DEFAULT_THRESHOLD = 0.01

_FONT_DIR = Path(__file__).resolve().parent
_SC_FONT = _FONT_DIR / "fonts/NotoSansCJK-SC-Regular.otf"


@dataclass
class EpisodeCurve:
    signal: str
    dim: int
    name: str
    episode_index: int
    frame_indices: np.ndarray
    timestamps: np.ndarray
    values: np.ndarray
    deltas: np.ndarray
    max_abs_delta: float
    max_open_delta: float
    max_close_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 LeRobot v2.1 数据集 state/action 夹爪相邻帧 delta。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        "--dataset-dir",
        type=Path,
        default=None,
        help="LeRobot v2.1 数据集目录。传入后优先使用该目录。",
    )
    parser.add_argument("--repo_id", type=str, default=None, help="LeRobot 数据集 repo id。")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="数据集 root。若 root/meta/info.json 存在，则 root 会被视为数据集目录。",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="输出 CSV/PDF 的目录。")
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="最多处理多少个 episode；不传或传 0 表示全量。",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default=",".join(str(dim) for dim in DEFAULT_DIMS),
        help="要统计的 0 基维度下标，支持逗号或空格分隔。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="异常跳变阈值，abs(delta) >= threshold 会写入 events CSV。",
    )
    parser.add_argument(
        "--top_episodes",
        "--top-episodes",
        type=int,
        default=3,
        help="PDF 中每个 signal/dim 展示的最大跳变 episode 数。",
    )
    return parser.parse_args()


def parse_dims(value: str) -> list[int]:
    dims: list[int] = []
    for part in re.split(r"[,\s]+", value.strip()):
        if not part:
            continue
        dim = int(part)
        if dim < 0:
            raise ValueError(f"--dims 只能包含非负 0 基下标，当前为 {dim}")
        dims.append(dim)
    if not dims:
        raise ValueError("--dims 至少需要包含一个维度")
    return list(dict.fromkeys(dims))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def format_dataset_path(template: str, episode_index: int, chunks_size: int) -> Path:
    values = {
        "episode_chunk": episode_chunk(episode_index, chunks_size),
        "episode_index": episode_index,
    }
    return Path(template.format(**values))


def looks_like_dataset_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").exists()


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir.expanduser()

    if args.root is not None:
        root = args.root.expanduser()
        if looks_like_dataset_dir(root):
            return root
        if args.repo_id:
            candidate = root / args.repo_id
            if looks_like_dataset_dir(candidate):
                return candidate
        return root

    if args.repo_id:
        repo_path = Path(args.repo_id).expanduser()
        if looks_like_dataset_dir(repo_path):
            return repo_path
        lerobot_home = os.environ.get("HF_LEROBOT_HOME") or os.environ.get("LEROBOT_HOME")
        if lerobot_home:
            return Path(lerobot_home).expanduser() / args.repo_id
        return Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id

    raise ValueError("必须提供 --dataset_dir，或同时提供可解析的数据集 --repo_id/--root")


def validate_dataset_dir(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "meta" / "info.json",
        dataset_dir / "meta" / "episodes.jsonl",
        dataset_dir / "data",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少 LeRobot v2.1 数据集文件/目录: " + ", ".join(str(p) for p in missing))


def sanitize_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "dataset"


def feature_names(info: dict[str, Any], signal: str) -> list[str]:
    names = info.get("features", {}).get(signal, {}).get("names")
    if not names:
        return []
    if isinstance(names, list) and names and isinstance(names[0], list):
        names = names[0]
    if not isinstance(names, list):
        return []
    return [str(name) for name in names]


def dim_name(info: dict[str, Any], signal: str, dim: int) -> str:
    names = feature_names(info, signal)
    if 0 <= dim < len(names):
        return names[dim]
    return f"dim_{dim}"


def numeric_array_from_series(series: pd.Series) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row_index, value in enumerate(series):
        if value is None:
            raise ValueError(f"第 {row_index} 行为空值")
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        rows.append(arr)

    if not rows:
        return np.empty((0, 0), dtype=np.float64)

    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError(f"向量维度不一致: {sorted(widths)}")
    return np.vstack(rows)


def scalar_from_cell(value: Any) -> float:
    if isinstance(value, np.ndarray):
        arr = value.reshape(-1)
        return float(arr[0]) if arr.size else math.nan
    if isinstance(value, (list, tuple)):
        return scalar_from_cell(np.asarray(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def scalar_column(df: pd.DataFrame, column: str, fallback: np.ndarray) -> np.ndarray:
    if column not in df.columns:
        return fallback.astype(np.float64)
    return np.asarray([scalar_from_cell(value) for value in df[column]], dtype=np.float64)


def format_frame(value: float) -> int | float | str:
    if not np.isfinite(value):
        return ""
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return float(value)


def direction(delta: float) -> str:
    if delta > 0:
        return "open"
    if delta < 0:
        return "close"
    return "zero"


def ordered_keys(signals: tuple[str, ...], dims: list[int]) -> list[tuple[str, int]]:
    keys = [(signal, dim) for signal in signals for dim in dims]

    def sort_key(item: tuple[str, int]) -> tuple[int, int, int]:
        signal, dim = item
        right_priority = 0 if dim == 13 else 1
        signal_priority = 0 if signal == "observation.state" else 1
        return (right_priority, signal_priority, dim)

    return sorted(keys, key=sort_key)


def add_top_curve(curves: list[EpisodeCurve], curve: EpisodeCurve, limit: int) -> None:
    if limit <= 0:
        return
    curves.append(curve)
    curves.sort(key=lambda item: item.max_abs_delta, reverse=True)
    del curves[limit:]


def analyze_dataset(
    dataset_dir: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    dims: list[int],
    threshold: float,
    top_episodes: int,
) -> tuple[
    dict[tuple[str, int], list[np.ndarray]],
    list[dict[str, Any]],
    dict[tuple[str, int], list[EpisodeCurve]],
    list[dict[str, Any]],
]:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    delta_parts: dict[tuple[str, int], list[np.ndarray]] = {
        key: [] for key in ordered_keys(SIGNALS, dims)
    }
    top_curves: dict[tuple[str, int], list[EpisodeCurve]] = {
        key: [] for key in ordered_keys(SIGNALS, dims)
    }
    event_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for episode in tqdm(episodes, desc="扫描 episode"):
        episode_index = int(episode.get("episode_index", len(skipped_rows)))
        parquet_path = dataset_dir / format_dataset_path(data_path, episode_index, chunks_size)
        if not parquet_path.exists():
            skipped_rows.append(
                {
                    "episode": episode_index,
                    "signal": "",
                    "reason": "parquet 不存在",
                    "path": str(parquet_path),
                }
            )
            continue

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            skipped_rows.append(
                {
                    "episode": episode_index,
                    "signal": "",
                    "reason": f"读取 parquet 失败: {exc}",
                    "path": str(parquet_path),
                }
            )
            continue

        expected_len = episode.get("length")
        if expected_len is not None and int(expected_len) != len(df):
            skipped_rows.append(
                {
                    "episode": episode_index,
                    "signal": "",
                    "reason": f"行数和 episodes.jsonl 不一致: parquet={len(df)}, metadata={expected_len}",
                    "path": str(parquet_path),
                }
            )

        if len(df) < 2:
            skipped_rows.append(
                {
                    "episode": episode_index,
                    "signal": "",
                    "reason": "episode 少于 2 帧，无法计算 delta",
                    "path": str(parquet_path),
                }
            )
            continue

        fallback_index = np.arange(len(df), dtype=np.float64)
        frame_indices = scalar_column(df, "frame_index", fallback_index)
        timestamps = scalar_column(df, "timestamp", np.full(len(df), np.nan, dtype=np.float64))

        for signal in SIGNALS:
            if signal not in df.columns:
                skipped_rows.append(
                    {
                        "episode": episode_index,
                        "signal": signal,
                        "reason": f"缺少列 {signal}",
                        "path": str(parquet_path),
                    }
                )
                continue

            try:
                values = numeric_array_from_series(df[signal])
            except Exception as exc:
                skipped_rows.append(
                    {
                        "episode": episode_index,
                        "signal": signal,
                        "reason": f"解析向量失败: {exc}",
                        "path": str(parquet_path),
                    }
                )
                continue

            if values.shape[0] != len(df):
                skipped_rows.append(
                    {
                        "episode": episode_index,
                        "signal": signal,
                        "reason": f"向量行数不一致: values={values.shape[0]}, rows={len(df)}",
                        "path": str(parquet_path),
                    }
                )
                continue

            for dim in dims:
                key = (signal, dim)
                name = dim_name(info, signal, dim)
                if dim >= values.shape[1]:
                    skipped_rows.append(
                        {
                            "episode": episode_index,
                            "signal": signal,
                            "reason": f"维度不足: dim={dim}, actual_dims={values.shape[1]}",
                            "path": str(parquet_path),
                        }
                    )
                    continue

                col = values[:, dim]
                deltas = col[1:] - col[:-1]
                if deltas.size == 0:
                    continue
                delta_parts[key].append(deltas)

                abs_deltas = np.abs(deltas)
                event_mask = abs_deltas >= threshold
                if np.any(event_mask):
                    for delta_pos in np.flatnonzero(event_mask):
                        row_index = int(delta_pos + 1)
                        delta = float(deltas[delta_pos])
                        event_rows.append(
                            {
                                "episode": episode_index,
                                "frame_index": format_frame(frame_indices[row_index]),
                                "timestamp": float(timestamps[row_index])
                                if np.isfinite(timestamps[row_index])
                                else "",
                                "signal": signal,
                                "dim": dim,
                                "name": name,
                                "prev": float(col[row_index - 1]),
                                "curr": float(col[row_index]),
                                "delta": delta,
                                "abs_delta": float(abs_deltas[delta_pos]),
                                "direction": direction(delta),
                            }
                        )

                max_abs = float(np.max(abs_deltas))
                existing = top_curves[key]
                should_store = len(existing) < top_episodes or (
                    top_episodes > 0 and max_abs > min(curve.max_abs_delta for curve in existing)
                )
                if should_store:
                    curve = EpisodeCurve(
                        signal=signal,
                        dim=dim,
                        name=name,
                        episode_index=episode_index,
                        frame_indices=frame_indices.copy(),
                        timestamps=timestamps.copy(),
                        values=col.copy(),
                        deltas=deltas.copy(),
                        max_abs_delta=max_abs,
                        max_open_delta=float(np.max(deltas)),
                        max_close_delta=float(np.min(deltas)),
                    )
                    add_top_curve(existing, curve, top_episodes)

    return delta_parts, event_rows, top_curves, skipped_rows


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, q))


def build_summary_rows(
    info: dict[str, Any],
    delta_parts: dict[tuple[str, int], list[np.ndarray]],
    event_rows: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    events_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in event_rows:
        events_by_key.setdefault((str(row["signal"]), int(row["dim"])), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, parts in delta_parts.items():
        signal, dim = key
        name = dim_name(info, signal, dim)
        deltas = np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)
        events = events_by_key.get(key, [])
        abnormal_episodes = {int(row["episode"]) for row in events}
        open_events = [row for row in events if float(row["delta"]) >= threshold]
        close_events = [row for row in events if float(row["delta"]) <= -threshold]

        if deltas.size:
            abs_values = np.abs(deltas)
            max_abs_index = int(np.argmax(abs_values))
            max_abs_delta = float(abs_values[max_abs_index])
            max_delta = float(deltas[max_abs_index])
        else:
            max_abs_delta = math.nan
            max_delta = math.nan

        summary_rows.append(
            {
                "signal": signal,
                "dim": dim,
                "name": name,
                "count": int(deltas.size),
                "mean": float(np.mean(deltas)) if deltas.size else math.nan,
                "std": float(np.std(deltas)) if deltas.size else math.nan,
                "min": float(np.min(deltas)) if deltas.size else math.nan,
                "max": float(np.max(deltas)) if deltas.size else math.nan,
                "p50": percentile(deltas, 50),
                "p95": percentile(deltas, 95),
                "p99": percentile(deltas, 99),
                "p99_9": percentile(deltas, 99.9),
                "abs_max": max_abs_delta,
                "max_abs_signed_delta": max_delta,
                "threshold": threshold,
                "abnormal_frames": len(events),
                "abnormal_ratio": len(events) / deltas.size if deltas.size else 0.0,
                "abnormal_episodes": len(abnormal_episodes),
                "open_frames": len(open_events),
                "open_episodes": len({int(row["episode"]) for row in open_events}),
                "close_frames": len(close_events),
                "close_episodes": len({int(row["episode"]) for row in close_events}),
            }
        )
    return summary_rows


def print_summary(summary_rows: list[dict[str, Any]], threshold: float) -> None:
    print(f"\n夹爪 delta 统计报告 (阈值: abs(delta) >= {threshold:g})")
    if not summary_rows:
        print("  未统计到任何 delta。")
        return

    header = (
        f"{'signal':<18} {'dim':>3} {'name':<20} {'count':>10} {'mean':>11} "
        f"{'std':>11} {'min':>11} {'max':>11} {'p50':>11} {'p95':>11} "
        f"{'p99':>11} {'p99.9':>11} {'abs_max':>11} {'events':>8} {'open':>8} {'eps':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['signal']:<18} {int(row['dim']):>3} {str(row['name']):<20.20} "
            f"{int(row['count']):>10} {float(row['mean']):>11.6f} {float(row['std']):>11.6f} "
            f"{float(row['min']):>11.6f} {float(row['max']):>11.6f} "
            f"{float(row['p50']):>11.6f} {float(row['p95']):>11.6f} "
            f"{float(row['p99']):>11.6f} "
            f"{float(row['p99_9']):>11.6f} {float(row['abs_max']):>11.6f} "
            f"{int(row['abnormal_frames']):>8} {int(row['open_frames']):>8} "
            f"{int(row['abnormal_episodes']):>6}"
        )


def setup_plot_style() -> None:
    for style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "default"):
        try:
            plt.style.use(style)
            break
        except Exception:
            continue
    if _SC_FONT.exists():
        fm.fontManager.addfont(str(_SC_FONT))
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "DejaVu Sans",
        "Arial",
        "SimHei",
        "WenQuanYi Micro Hei",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def add_summary_page(pdf: PdfPages, summary_rows: list[dict[str, Any]], skipped_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    lines = ["Gripper Delta Summary", ""]
    for row in summary_rows:
        lines.append(
            f"{row['signal']} dim {row['dim']} {row['name']}: "
            f"count={row['count']}, min={row['min']:.6f}, max={row['max']:.6f}, "
            f"p99.9={row['p99_9']:.6f}, abs_max={row['abs_max']:.6f}, "
            f"events={row['abnormal_frames']}, open={row['open_frames']}, episodes={row['abnormal_episodes']}"
        )
    if skipped_rows:
        lines.extend(["", f"Warnings / skipped issues: {len(skipped_rows)}"])
        for row in skipped_rows[:12]:
            lines.append(f"episode {row['episode']} {row['signal']}: {row['reason']}")
        if len(skipped_rows) > 12:
            lines.append(f"... only showing first 12 of {len(skipped_rows)} issues")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def add_distribution_page(
    pdf: PdfPages,
    delta_parts: dict[tuple[str, int], list[np.ndarray]],
    info: dict[str, Any],
    threshold: float,
) -> None:
    keys = [key for key, parts in delta_parts.items() if parts]
    if not keys:
        return

    n_cols = 2
    n_rows = math.ceil(len(keys) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(4, n_rows * 3.4)))
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    fig.suptitle("Delta Distribution", fontsize=14, fontweight="bold")

    for i, key in enumerate(keys):
        signal, dim = key
        ax = axes[i // n_cols, i % n_cols]
        values = np.concatenate(delta_parts[key])
        bins = min(120, max(20, int(np.sqrt(values.size))))
        ax.hist(values, bins=bins, color="#5B9BD5", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(threshold, color="#CD5C5C", linestyle="--", linewidth=1)
        ax.axvline(-threshold, color="#CD5C5C", linestyle="--", linewidth=1)
        if values.size > 100:
            ax.set_yscale("log")
        ax.set_title(f"{signal} dim {dim}: {dim_name(info, signal, dim)}", fontsize=10)
        ax.set_xlabel("delta")
        ax.set_ylabel("count")

    for i in range(len(keys), n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def add_curve_pages(
    pdf: PdfPages,
    top_curves: dict[tuple[str, int], list[EpisodeCurve]],
    threshold: float,
) -> None:
    for key, curves in top_curves.items():
        if not curves:
            continue
        signal, dim = key
        n = len(curves)
        fig, axes = plt.subplots(n, 1, figsize=(12, max(3.2, n * 3.0)), sharex=False)
        if n == 1:
            axes = np.asarray([axes])
        fig.suptitle(f"Top Episodes · {signal} dim {dim}: {curves[0].name}", fontsize=14, fontweight="bold")

        for ax, curve in zip(axes, curves, strict=True):
            x_values = curve.frame_indices
            x_delta = curve.frame_indices[1:]
            ax.plot(x_values, curve.values, color="#5B9BD5", linewidth=1.1, label="value")
            ax.set_ylabel("value")
            ax2 = ax.twinx()
            ax2.plot(x_delta, curve.deltas, color="#ED7D31", linewidth=0.8, alpha=0.85, label="delta")
            ax2.axhline(threshold, color="#CD5C5C", linestyle="--", linewidth=0.8)
            ax2.axhline(-threshold, color="#CD5C5C", linestyle="--", linewidth=0.8)
            event_mask = np.abs(curve.deltas) >= threshold
            if np.any(event_mask):
                ax2.scatter(
                    x_delta[event_mask],
                    curve.deltas[event_mask],
                    s=18,
                    color="#CD5C5C",
                    zorder=5,
                    label="event",
                )
            ax2.set_ylabel("delta")
            ax.set_title(
                f"episode {curve.episode_index} | abs_max={curve.max_abs_delta:.6f} | "
                f"max_open={curve.max_open_delta:.6f} | max_close={curve.max_close_delta:.6f}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

        axes[-1].set_xlabel("frame_index")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig, dpi=200)
        plt.close(fig)


def write_pdf_report(
    output_path: Path,
    info: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    delta_parts: dict[tuple[str, int], list[np.ndarray]],
    top_curves: dict[tuple[str, int], list[EpisodeCurve]],
    skipped_rows: list[dict[str, Any]],
    threshold: float,
) -> None:
    setup_plot_style()
    with warnings.catch_warnings(), PdfPages(output_path) as pdf:
        warnings.simplefilter("ignore", UserWarning)
        add_summary_page(pdf, summary_rows, skipped_rows)
        add_distribution_page(pdf, delta_parts, info, threshold)
        add_curve_pages(pdf, top_curves, threshold)


def main() -> None:
    args = parse_args()
    dims = parse_dims(args.dims)
    if args.threshold < 0:
        raise ValueError("--threshold 不能为负数")
    if args.top_episodes < 0:
        raise ValueError("--top_episodes 不能为负数")

    dataset_dir = resolve_dataset_dir(args).resolve()
    validate_dataset_dir(dataset_dir)
    info = read_json(dataset_dir / "meta" / "info.json")
    episodes = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    episodes.sort(key=lambda row: int(row.get("episode_index", 0)))

    if args.max_episodes is not None and args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    output_dir = (args.output_dir or Path.cwd()).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = sanitize_name(dataset_dir.name)

    print("LeRobot 夹爪 delta 诊断")
    print(f"  dataset_dir : {dataset_dir}")
    print(f"  episodes    : {len(episodes)}")
    print(f"  dims        : {dims}")
    print(f"  threshold   : {args.threshold:g}")
    for signal in SIGNALS:
        names = feature_names(info, signal)
        selected_names = [names[dim] if dim < len(names) else f"dim_{dim}" for dim in dims]
        print(f"  {signal:<17}: {selected_names}")

    delta_parts, event_rows, top_curves, skipped_rows = analyze_dataset(
        dataset_dir=dataset_dir,
        info=info,
        episodes=episodes,
        dims=dims,
        threshold=args.threshold,
        top_episodes=args.top_episodes,
    )
    summary_rows = build_summary_rows(info, delta_parts, event_rows, args.threshold)
    print_summary(summary_rows, args.threshold)

    if skipped_rows:
        print(f"\n⚠️  扫描过程中记录了 {len(skipped_rows)} 个 warning/skip，前 10 个如下：")
        for row in skipped_rows[:10]:
            print(f"  - episode {row['episode']} {row['signal']}: {row['reason']}")
        if len(skipped_rows) > 10:
            print(f"    ... 其余 {len(skipped_rows) - 10} 个未在终端展开")

    event_columns = [
        "episode",
        "frame_index",
        "timestamp",
        "signal",
        "dim",
        "name",
        "prev",
        "curr",
        "delta",
        "abs_delta",
        "direction",
    ]
    summary_path = output_dir / f"gripper_delta_summary_{dataset_name}.csv"
    events_path = output_dir / f"gripper_delta_events_{dataset_name}.csv"
    pdf_path = output_dir / f"gripper_delta_report_{dataset_name}.pdf"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(event_rows, columns=event_columns).to_csv(events_path, index=False)
    write_pdf_report(
        output_path=pdf_path,
        info=info,
        summary_rows=summary_rows,
        delta_parts=delta_parts,
        top_curves=top_curves,
        skipped_rows=skipped_rows,
        threshold=args.threshold,
    )

    print("\n输出文件：")
    print(f"  summary CSV : {summary_path}")
    print(f"  events CSV  : {events_path}")
    print(f"  PDF report  : {pdf_path}")
    print(f"\n超过阈值的事件数: {len(event_rows)}")


if __name__ == "__main__":
    main()
