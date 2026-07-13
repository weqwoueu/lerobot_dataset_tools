#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查找左臂在指定位置附近连续静止过久的 LeRobot v2.1 episodes。

脚本只读取数据集 parquet，不修改原始数据。默认检查 ``observation.state``
前 7 维（左臂 6 个关节和左夹爪），右臂是否运动不影响判定。

示例：
    python 24_find_left_pose_idle_episodes.py

    python 24_find_left_pose_idle_episodes.py \
        --dataset_dir /path/to/lerobot_dataset \
        --joint_position_tolerance 0.1 \
        --min_duration_seconds 3.0 \
        --output_json ./output/left_pose_idle_episodes.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DATASET_DIR = Path(
    "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0703_nonidle"
)
DEFAULT_TARGET_STATE = (
    -0.9036,
    1.3299,
    -1.0784,
    0.6839,
    0.3320,
    -0.0663,
    0.0307,
    0.1975,
    2.0976,
    -1.8200,
    0.7614,
    0.0550,
    -0.0606,
    0.0084,
)
EXPECTED_LEFT_NAMES = (
    "left_joint_1_pos",
    "left_joint_2_pos",
    "left_joint_3_pos",
    "left_joint_4_pos",
    "left_joint_5_pos",
    "left_joint_6_pos",
    "left_gripper_pos",
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdleSegment:
    """一个满足目标位置、静止阈值和时长要求的半开区间。"""

    start: int
    end: int
    frame_count: int
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="查找左臂在指定位置附近连续静止超过阈值的 episode。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="LeRobot v2.1 数据集目录。脚本只读，不修改源数据。",
    )
    parser.add_argument(
        "--state_key",
        "--state-key",
        default="observation.state",
        help="parquet 中的 state 列名。",
    )
    parser.add_argument(
        "--target_state",
        "--target-state",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_STATE),
        help="目标 state，可传 7 维左臂状态或完整 14 维状态；只使用前 7 维。",
    )
    parser.add_argument(
        "--joint_position_tolerance",
        "--joint-position-tolerance",
        type=float,
        default=0.15,
        help="左臂 6 个关节相对目标位置的逐维绝对误差上限（rad）。",
    )
    parser.add_argument(
        "--gripper_position_tolerance",
        "--gripper-position-tolerance",
        type=float,
        default=0.01,
        help="左夹爪相对目标位置的绝对误差上限。",
    )
    parser.add_argument(
        "--stationary_tolerance",
        "--stationary-tolerance",
        type=float,
        default=0.001,
        help="左臂 7 维相邻帧变化的逐维绝对值上限。",
    )
    parser.add_argument(
        "--min_duration_seconds",
        "--min-duration-seconds",
        type=float,
        default=2.0,
        help="最短静止时间；只有严格超过该秒数的区间才命中。",
    )
    parser.add_argument(
        "--output_json",
        "--output-json",
        type=Path,
        default=None,
        help="诊断 JSON 路径。默认写入脚本目录下的 output/。",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
            rows.append(value)
    return rows


def validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是大于 0 的有限数，当前为 {value}")


def validate_dataset_dir(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "meta" / "info.json",
        dataset_dir / "meta" / "episodes.jsonl",
        dataset_dir / "data",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少 LeRobot v2.1 数据集文件/目录: " + ", ".join(str(path) for path in missing)
        )


def validate_target_state(values: list[float]) -> np.ndarray:
    target = np.asarray(values, dtype=np.float64)
    if target.ndim != 1 or target.size not in {7, 14}:
        raise ValueError(f"--target_state 必须包含 7 或 14 个数，当前为 {target.size} 个")
    if not np.all(np.isfinite(target)):
        raise ValueError("--target_state 不能包含 NaN 或无穷值")
    return target


def validate_state_feature(info: dict[str, Any], state_key: str) -> None:
    feature = info.get("features", {}).get(state_key)
    if not isinstance(feature, dict):
        raise KeyError(f"meta/info.json 的 features 中不存在 {state_key}")

    shape = feature.get("shape")
    if not isinstance(shape, list) or not shape or int(shape[0]) < 7:
        raise ValueError(f"{state_key} 的 feature shape 至少需要 7 维，当前为 {shape}")

    names = feature.get("names")
    if names is not None:
        if not isinstance(names, list) or len(names) < 7:
            raise ValueError(f"{state_key} 的 feature names 少于 7 项")
        actual_left_names = tuple(str(name) for name in names[:7])
        if actual_left_names != EXPECTED_LEFT_NAMES:
            raise ValueError(
                f"{state_key} 前 7 维名称不符合左臂顺序: {actual_left_names}; 预期为 {EXPECTED_LEFT_NAMES}"
            )


def format_dataset_path(template: str, episode_index: int, chunks_size: int) -> Path:
    return Path(
        template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
    )


def numeric_array_from_series(series: pd.Series) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row_index, value in enumerate(series):
        if value is None:
            raise ValueError(f"state 第 {row_index} 行为空值")
        row = np.asarray(value, dtype=np.float64).reshape(-1)
        if row.size < 7:
            raise ValueError(f"state 第 {row_index} 行只有 {row.size} 维，至少需要 7 维")
        if not np.all(np.isfinite(row[:7])):
            raise ValueError(f"state 第 {row_index} 行左臂状态包含 NaN 或无穷值")
        rows.append(row)

    if not rows:
        return np.empty((0, 0), dtype=np.float64)

    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError(f"state 各行维度不一致: {sorted(widths)}")
    return np.vstack(rows)


def contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"mask 必须是一维，当前 shape={mask.shape}")
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def find_qualifying_segments(
    states: np.ndarray,
    target_left: np.ndarray,
    fps: float,
    joint_position_tolerance: float,
    gripper_position_tolerance: float,
    stationary_tolerance: float,
    min_duration_seconds: float,
) -> list[IdleSegment]:
    """返回左臂在目标附近静止且持续时间严格超过阈值的区间。"""

    states = np.asarray(states, dtype=np.float64)
    target_left = np.asarray(target_left, dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] < 7:
        raise ValueError(f"states 必须是至少 7 维的二维数组，当前 shape={states.shape}")
    if target_left.size != 7:
        raise ValueError(f"target_left 必须是 7 维，当前为 {target_left.size} 维")
    if states.shape[0] == 0:
        return []

    left_states = states[:, :7]
    position_error = np.abs(left_states - target_left)
    near_target = np.all(position_error[:, :6] <= joint_position_tolerance, axis=1)
    near_target &= position_error[:, 6] <= gripper_position_tolerance

    stationary = np.ones(states.shape[0], dtype=bool)
    if states.shape[0] > 1:
        stationary[1:] = np.all(np.abs(np.diff(left_states, axis=0)) <= stationary_tolerance, axis=1)

    segments: list[IdleSegment] = []
    for start, end in contiguous_true_ranges(near_target & stationary):
        frame_count = end - start
        duration_seconds = frame_count / fps
        if duration_seconds > min_duration_seconds:
            segments.append(
                IdleSegment(
                    start=start,
                    end=end,
                    frame_count=frame_count,
                    duration_seconds=duration_seconds,
                )
            )
    return segments


def scalar_at(df: pd.DataFrame, column: str, row: int, fallback: float) -> float:
    if column not in df.columns:
        return fallback
    value = np.asarray(df[column].iloc[row]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"{column} 第 {row} 行不是标量")
    result = float(value[0])
    if not math.isfinite(result):
        raise ValueError(f"{column} 第 {row} 行不是有限数")
    return result


def display_number(value: float) -> int | float:
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def segment_diagnostics(
    df: pd.DataFrame,
    states: np.ndarray,
    target_left: np.ndarray,
    segment: IdleSegment,
    fps: float,
) -> dict[str, Any]:
    start = segment.start
    end_inclusive = segment.end - 1
    left_segment = states[start : segment.end, :7]
    position_error = np.abs(left_segment - target_left)
    deltas = np.abs(np.diff(left_segment, axis=0))
    max_stationary_delta = float(np.max(deltas)) if deltas.size else 0.0

    start_frame = scalar_at(df, "frame_index", start, float(start))
    end_frame = scalar_at(df, "frame_index", end_inclusive, float(end_inclusive))
    start_timestamp = scalar_at(df, "timestamp", start, start / fps)
    end_timestamp = scalar_at(df, "timestamp", end_inclusive, end_inclusive / fps)

    return {
        "start_row": start,
        "end_row": end_inclusive,
        "start_frame_index": display_number(start_frame),
        "end_frame_index": display_number(end_frame),
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "frame_count": segment.frame_count,
        "duration_seconds": segment.duration_seconds,
        "max_joint_position_error": float(np.max(position_error[:, :6])),
        "max_gripper_position_error": float(np.max(position_error[:, 6])),
        "max_stationary_delta": max_stationary_delta,
    }


def resolve_output_json(dataset_dir: Path, configured_path: Path | None) -> Path:
    if configured_path is not None:
        return configured_path.expanduser().resolve()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_dir.name).strip("_")
    filename = f"left_pose_idle_episodes_{safe_name or 'dataset'}.json"
    return Path(__file__).resolve().parent / "output" / filename


def analyze_dataset(
    dataset_dir: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    state_key: str,
    target_left: np.ndarray,
    joint_position_tolerance: float,
    gripper_position_tolerance: float,
    stationary_tolerance: float,
    min_duration_seconds: float,
) -> list[dict[str, Any]]:
    fps = float(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    if chunks_size <= 0:
        raise ValueError(f"meta/info.json 的 chunks_size 必须大于 0，当前为 {chunks_size}")
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("meta/info.json 的 data_path 必须是非空字符串")

    matches: list[dict[str, Any]] = []
    for position, episode in enumerate(episodes, start=1):
        if "episode_index" not in episode:
            raise ValueError(f"episodes.jsonl 第 {position} 行缺少 episode_index")
        episode_index = int(episode["episode_index"])
        parquet_path = dataset_dir / format_dataset_path(data_path, episode_index, chunks_size)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"episode {episode_index} 的 parquet 不存在: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        if state_key not in df.columns:
            raise KeyError(f"episode {episode_index} 的 parquet 缺少 {state_key}; 实际列: {list(df.columns)}")
        expected_length = episode.get("length")
        if expected_length is not None and len(df) != int(expected_length):
            raise ValueError(
                f"episode {episode_index} 行数与元数据不一致: "
                f"parquet={len(df)}, episodes.jsonl={expected_length}"
            )
        if "episode_index" in df.columns:
            parquet_episode_indices = np.asarray(df["episode_index"], dtype=np.int64)
            if parquet_episode_indices.size and not np.all(parquet_episode_indices == episode_index):
                raise ValueError(f"episode {episode_index} 的 parquet 包含不一致的 episode_index")

        states = numeric_array_from_series(df[state_key])
        segments = find_qualifying_segments(
            states=states,
            target_left=target_left,
            fps=fps,
            joint_position_tolerance=joint_position_tolerance,
            gripper_position_tolerance=gripper_position_tolerance,
            stationary_tolerance=stationary_tolerance,
            min_duration_seconds=min_duration_seconds,
        )
        if segments:
            longest = max(segments, key=lambda item: (item.frame_count, -item.start))
            matches.append(
                {
                    "episode_index": episode_index,
                    "qualifying_segment_count": len(segments),
                    "longest_segment": segment_diagnostics(
                        df=df,
                        states=states,
                        target_left=target_left,
                        segment=longest,
                        fps=fps,
                    ),
                }
            )

        if position % 100 == 0 or position == len(episodes):
            logger.info("已扫描 %d/%d episodes，当前命中 %d 个", position, len(episodes), len(matches))

    return sorted(matches, key=lambda item: int(item["episode_index"]))


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    validate_positive("--joint_position_tolerance", args.joint_position_tolerance)
    validate_positive("--gripper_position_tolerance", args.gripper_position_tolerance)
    validate_positive("--stationary_tolerance", args.stationary_tolerance)
    validate_positive("--min_duration_seconds", args.min_duration_seconds)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    validate_dataset_dir(dataset_dir)
    info = read_json(dataset_dir / "meta" / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"仅支持 LeRobot v2.1，当前 codebase_version={info.get('codebase_version')!r}")
    if "fps" not in info:
        raise ValueError("meta/info.json 缺少 fps")
    fps = float(info["fps"])
    validate_positive("meta/info.json fps", fps)
    validate_state_feature(info, args.state_key)

    target_state = validate_target_state(args.target_state)
    target_left = target_state[:7]
    episodes = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    if not episodes:
        raise ValueError("meta/episodes.jsonl 为空")
    episode_indices = [int(episode["episode_index"]) for episode in episodes if "episode_index" in episode]
    if len(episode_indices) != len(episodes):
        raise ValueError("meta/episodes.jsonl 存在缺少 episode_index 的记录")
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("meta/episodes.jsonl 存在重复的 episode_index")
    total_episodes = info.get("total_episodes")
    if total_episodes is not None and int(total_episodes) != len(episodes):
        raise ValueError(f"episode 数量与元数据不一致: episodes.jsonl={len(episodes)}, info={total_episodes}")

    logger.info("数据集: %s", dataset_dir)
    logger.info(
        "规则: joint_tol=%g, gripper_tol=%g, stationary_tol=%g, duration>%gs, fps=%g",
        args.joint_position_tolerance,
        args.gripper_position_tolerance,
        args.stationary_tolerance,
        args.min_duration_seconds,
        fps,
    )
    matches = analyze_dataset(
        dataset_dir=dataset_dir,
        info=info,
        episodes=episodes,
        state_key=args.state_key,
        target_left=target_left,
        joint_position_tolerance=args.joint_position_tolerance,
        gripper_position_tolerance=args.gripper_position_tolerance,
        stationary_tolerance=args.stationary_tolerance,
        min_duration_seconds=args.min_duration_seconds,
    )
    matched_indices = [int(match["episode_index"]) for match in matches]

    output_json = resolve_output_json(dataset_dir, args.output_json)
    report = {
        "schema_version": 1,
        "dataset_dir": str(dataset_dir),
        "state_key": args.state_key,
        "fps": fps,
        "target_state": target_state.tolist(),
        "left_target_state": target_left.tolist(),
        "thresholds": {
            "joint_position_tolerance": args.joint_position_tolerance,
            "gripper_position_tolerance": args.gripper_position_tolerance,
            "stationary_tolerance": args.stationary_tolerance,
            "min_duration_seconds_exclusive": args.min_duration_seconds,
        },
        "total_episode_count": len(episodes),
        "matched_episode_count": len(matches),
        "episode_indices": matched_indices,
        "matches": matches,
    }
    write_report(output_json, report)

    print(f"命中 episode 数量: {len(matched_indices)}")
    for match in matches:
        segment = match["longest_segment"]
        print(
            f"episode {match['episode_index']}: "
            f"frames {segment['start_frame_index']}-{segment['end_frame_index']}, "
            f"duration={segment['duration_seconds']:.3f}s, "
            f"joint_error<={segment['max_joint_position_error']:.6f}, "
            f"gripper_error<={segment['max_gripper_position_error']:.6f}"
        )
    print(f"episode_indices = {matched_indices}")
    print(f"诊断 JSON: {output_json}")


if __name__ == "__main__":
    main()
