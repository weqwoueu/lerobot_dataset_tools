#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Piper LeRobot v2.1 quaternion TCP state to 10-d rot6d model state.

The source dataset is never modified. By default this script reads the local
Piper peg-insertion non-idle dataset and writes a sibling dataset suffixed with
``_tcp_rot6d``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled in require_pandas()
    pd = None


def find_repo_root() -> Path:
    env_root = os.environ.get("RLINF_REPO_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([Path.cwd(), Path(__file__).absolute(), Path(__file__).resolve()])

    for candidate in candidates:
        start = candidate if candidate.is_dir() else candidate.parent
        for parent in (start, *start.parents):
            if (parent / "rlinf/utils/rot6d.py").is_file():
                return parent
            sibling = parent / "RLinf"
            if (sibling / "rlinf/utils/rot6d.py").is_file():
                return sibling

    raise RuntimeError(
        "Could not locate RLinf repo root. Set RLINF_REPO_ROOT or run from the RLinf repository."
    )


def load_quat_xyzw_to_rot6d(repo_root: Path) -> Any:
    module_path = repo_root / "rlinf/utils/rot6d.py"
    spec = importlib.util.spec_from_file_location("rlinf_rot6d_for_dataset_tool", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rot6d helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.quat_xyzw_to_rot6d


REPO_ROOT = find_repo_root()
quat_xyzw_to_rot6d = load_quat_xyzw_to_rot6d(REPO_ROOT)


DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "temp/dataset/piper_peg_insertion_rot6d/lerobot/piper_peg_insertion_rot6d_nonidle"
)
STATE_KEY_CANDIDATES = ("observation.state", "state")
REQUIRED_SOURCE_STATE_NAMES = (
    "gripper_position",
    "tcp_pose_x",
    "tcp_pose_y",
    "tcp_pose_z",
    "tcp_pose_qx",
    "tcp_pose_qy",
    "tcp_pose_qz",
    "tcp_pose_qw",
)
ROT6D_STATE_NAMES = [
    "tcp_pose_x",
    "tcp_pose_y",
    "tcp_pose_z",
    "tcp_pose_rot6d_col0_x",
    "tcp_pose_rot6d_col0_y",
    "tcp_pose_rot6d_col0_z",
    "tcp_pose_rot6d_col1_x",
    "tcp_pose_rot6d_col1_y",
    "tcp_pose_rot6d_col1_z",
    "gripper_position",
]
CORE_META_FILES = {"info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Piper LeRobot v2.1 quaternion state to 10-d rot6d model state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Source LeRobot v2.1 dataset directory. It will not be modified.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output dataset directory. Defaults to a sibling named <dataset>_tcp_rot6d.",
    )
    parser.add_argument(
        "--state_key",
        default=None,
        help="State column/feature key. Defaults to auto-detecting observation.state or state.",
    )
    parser.add_argument(
        "--output_state_key",
        default=None,
        help="Output state column/feature key. Defaults to the input state key.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and print the planned conversion without creating files.",
    )
    parser.add_argument(
        "--video_mode",
        choices=("hardlink", "copy", "symlink", "skip"),
        default="hardlink",
        help="How to place source videos in the output dataset. hardlink falls back to copy.",
    )
    return parser.parse_args()


def require_pandas() -> Any:
    if pd is None:
        raise RuntimeError("Missing pandas. Please run this in the LeRobot data-processing environment.")
    return pd


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def format_dataset_path(template: str, episode_index: int, chunks_size: int) -> Path:
    return Path(
        template.format(
            episode_chunk=episode_chunk(episode_index, chunks_size),
            episode_index=episode_index,
        )
    )


def validate_dataset_dir(dataset_dir: Path) -> None:
    required_paths = [
        dataset_dir / "meta" / "info.json",
        dataset_dir / "meta" / "episodes.jsonl",
        dataset_dir / "meta" / "tasks.jsonl",
        dataset_dir / "data",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required LeRobot v2.1 dataset paths: " + ", ".join(str(path) for path in missing)
        )


def resolve_output_dir(dataset_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return dataset_dir.with_name(f"{dataset_dir.name}_tcp_rot6d")


def resolve_state_key(info: dict[str, Any], requested_key: str | None) -> str:
    features = info.get("features", {})
    if requested_key is not None:
        if requested_key not in features:
            raise KeyError(f"State key not found in info.json features: {requested_key}")
        return requested_key

    for key in STATE_KEY_CANDIDATES:
        if key in features:
            return key
    raise KeyError(
        "Could not auto-detect state key. Expected one of: " + ", ".join(STATE_KEY_CANDIDATES)
    )


def get_state_indices(state_feature: dict[str, Any]) -> dict[str, int]:
    names = state_feature.get("names")
    if not isinstance(names, list):
        raise ValueError("State feature must contain a names list.")

    missing = [name for name in REQUIRED_SOURCE_STATE_NAMES if name not in names]
    if missing:
        raise ValueError("State feature is missing required names: " + ", ".join(missing))

    return {name: int(names.index(name)) for name in REQUIRED_SOURCE_STATE_NAMES}


def get_expected_parquet_paths(info: dict[str, Any], episodes: list[dict[str, Any]]) -> list[Path]:
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunks_size = int(info.get("chunks_size", 1000))
    return [
        format_dataset_path(data_path, int(row["episode_index"]), chunks_size)
        for row in episodes
    ]


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not force:
        raise FileExistsError(f"Output directory already exists. Use --force to overwrite: {output_dir}")
    logger.warning("Removing existing output directory: %s", output_dir)
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def numeric_array_from_series(series: pd.Series) -> np.ndarray:
    if len(series) == 0:
        return np.asarray([])
    first_valid = next((item for item in series if item is not None), None)
    if isinstance(first_valid, np.ndarray):
        return np.stack(series.to_numpy())
    if isinstance(first_valid, (list, tuple)):
        return np.asarray(series.to_list())
    return series.to_numpy()


def stats_to_jsonable(values: np.ndarray) -> dict[str, list[Any]]:
    if values.ndim == 1:
        axes: int | tuple[int, ...] = 0
        keepdims = True
    else:
        axes = 0
        keepdims = False
    return {
        "min": np.min(values, axis=axes, keepdims=keepdims).tolist(),
        "max": np.max(values, axis=axes, keepdims=keepdims).tolist(),
        "mean": np.mean(values, axis=axes, keepdims=keepdims).tolist(),
        "std": np.std(values, axis=axes, keepdims=keepdims).tolist(),
        "count": [int(len(values))],
    }


def transform_state_values(states: np.ndarray, state_indices: dict[str, int], context: str) -> np.ndarray:
    if states.ndim != 2:
        raise ValueError(f"{context}: expected state array with shape (N, D), got {states.shape}")

    max_index = max(state_indices.values())
    if states.shape[1] <= max_index:
        raise ValueError(
            f"{context}: state dim {states.shape[1]} is too small for required index {max_index}"
        )

    quat = states[
        :,
        [
            state_indices["tcp_pose_qx"],
            state_indices["tcp_pose_qy"],
            state_indices["tcp_pose_qz"],
            state_indices["tcp_pose_qw"],
        ],
    ].astype(np.float32)
    quat_norm = np.linalg.norm(quat, axis=1)
    if not np.all(np.isfinite(quat)):
        bad = np.flatnonzero(~np.all(np.isfinite(quat), axis=1))[:10].tolist()
        raise ValueError(f"{context}: quaternion contains non-finite values at rows {bad}")
    if np.any(quat_norm < 1e-8):
        bad = np.flatnonzero(quat_norm < 1e-8)[:10].tolist()
        raise ValueError(f"{context}: quaternion norm is zero/near-zero at rows {bad}")

    xyz = states[
        :,
        [
            state_indices["tcp_pose_x"],
            state_indices["tcp_pose_y"],
            state_indices["tcp_pose_z"],
        ],
    ].astype(np.float32)
    gripper = states[:, [state_indices["gripper_position"]]].astype(np.float32)
    rot6d = quat_xyzw_to_rot6d(quat).astype(np.float32)
    return np.concatenate([xyz, rot6d, gripper], axis=1).astype(np.float32)


def replace_state_column(
    df: pd.DataFrame,
    state_key: str,
    output_state_key: str,
    transformed_states: np.ndarray,
) -> pd.DataFrame:
    state_pos = list(df.columns).index(state_key)
    state_series = pd.Series(
        [row.copy() for row in transformed_states],
        index=df.index,
        dtype=object,
    )

    if output_state_key == state_key:
        df = df.copy()
        df[state_key] = state_series
        return df

    if output_state_key in df.columns:
        raise ValueError(
            f"Output state key already exists in parquet columns: {output_state_key}"
        )

    columns = list(df.columns)
    df = df.drop(columns=[state_key]).copy()
    df.insert(state_pos, output_state_key, state_series)
    expected_columns = columns.copy()
    expected_columns[state_pos] = output_state_key
    return df[expected_columns]


def process_episode_parquet(
    dataset_dir: Path,
    output_dir: Path,
    relative_path: Path,
    state_key: str,
    output_state_key: str,
    state_indices: dict[str, int],
) -> dict[str, list[Any]]:
    pandas = require_pandas()
    src_path = dataset_dir / relative_path
    dst_path = output_dir / relative_path
    if not src_path.exists():
        raise FileNotFoundError(f"Missing source parquet: {src_path}")

    df = pandas.read_parquet(src_path)
    if state_key not in df.columns:
        raise KeyError(f"{src_path}: state column not found: {state_key}")

    states = numeric_array_from_series(df[state_key]).astype(np.float32)
    transformed_states = transform_state_values(states, state_indices, str(relative_path))
    out_df = replace_state_column(df, state_key, output_state_key, transformed_states)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(dst_path, index=False)
    return stats_to_jsonable(transformed_states)


def update_info(
    info: dict[str, Any],
    state_key: str,
    output_state_key: str,
    total_videos: int | None,
) -> dict[str, Any]:
    updated = dict(info)
    features = dict(updated.get("features", {}))
    state_feature = dict(features[state_key])
    state_feature["shape"] = [len(ROT6D_STATE_NAMES)]
    state_feature["names"] = ROT6D_STATE_NAMES

    if output_state_key == state_key:
        features[state_key] = state_feature
    else:
        if output_state_key in features:
            raise ValueError(f"Output state key already exists in info.json features: {output_state_key}")
        ordered_features: dict[str, Any] = {}
        for key, feature in features.items():
            if key == state_key:
                ordered_features[output_state_key] = state_feature
            else:
                ordered_features[key] = feature
        features = ordered_features

    updated["features"] = features
    if total_videos is not None:
        updated["total_videos"] = total_videos
    return updated


def copy_meta_files(
    dataset_dir: Path,
    output_dir: Path,
    updated_info: dict[str, Any],
    updated_episode_stats: list[dict[str, Any]],
) -> None:
    src_meta_dir = dataset_dir / "meta"
    dst_meta_dir = output_dir / "meta"
    dst_meta_dir.mkdir(parents=True, exist_ok=True)

    for src_path in sorted(src_meta_dir.iterdir()):
        if src_path.is_file() and src_path.name not in CORE_META_FILES:
            shutil.copy2(src_path, dst_meta_dir / src_path.name)

    write_json(dst_meta_dir / "info.json", updated_info)
    shutil.copy2(src_meta_dir / "episodes.jsonl", dst_meta_dir / "episodes.jsonl")
    shutil.copy2(src_meta_dir / "tasks.jsonl", dst_meta_dir / "tasks.jsonl")
    write_jsonl(dst_meta_dir / "episodes_stats.jsonl", updated_episode_stats)


def build_updated_episode_stats(
    dataset_dir: Path,
    episodes: list[dict[str, Any]],
    state_key: str,
    output_state_key: str,
    state_stats_by_episode: dict[int, dict[str, list[Any]]],
) -> list[dict[str, Any]]:
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    previous_rows = read_jsonl(stats_path) if stats_path.exists() else []
    previous_by_episode = {
        int(row["episode_index"]): row
        for row in previous_rows
        if "episode_index" in row
    }

    updated_rows = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        row = dict(previous_by_episode.get(episode_index, {"episode_index": episode_index, "stats": {}}))
        stats = dict(row.get("stats", {}))
        if output_state_key != state_key:
            stats.pop(state_key, None)
        stats[output_state_key] = state_stats_by_episode[episode_index]
        row["episode_index"] = episode_index
        row["stats"] = stats
        updated_rows.append(row)
    return updated_rows


def place_videos(dataset_dir: Path, output_dir: Path, video_mode: str) -> int | None:
    if video_mode == "skip":
        return 0

    src_video_dir = dataset_dir / "videos"
    if not src_video_dir.exists():
        logger.info("Source dataset has no videos directory; skipping videos.")
        return 0

    video_files = sorted(path for path in src_video_dir.rglob("*") if path.is_file())
    for src_path in video_files:
        relative_path = src_path.relative_to(dataset_dir)
        dst_path = output_dir / relative_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if video_mode == "copy":
            shutil.copy2(src_path, dst_path)
        elif video_mode == "symlink":
            os.symlink(src_path.resolve(), dst_path)
        elif video_mode == "hardlink":
            try:
                os.link(src_path, dst_path)
            except OSError:
                shutil.copy2(src_path, dst_path)
        else:  # pragma: no cover - argparse enforces choices.
            raise ValueError(f"Unsupported video mode: {video_mode}")

    return len(video_files)


def write_readme(
    output_dir: Path,
    dataset_dir: Path,
    state_key: str,
    output_state_key: str,
) -> None:
    readme = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- rot6d
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# Piper Peg Insertion TCP Rot6D Dataset

This dataset was generated from:

`{dataset_dir}`

The `{state_key}` feature was converted to `{output_state_key}` with layout:

`[tcp_pose_x, tcp_pose_y, tcp_pose_z, tcp_pose_rot6d_col0_x, tcp_pose_rot6d_col0_y, tcp_pose_rot6d_col0_z, tcp_pose_rot6d_col1_x, tcp_pose_rot6d_col1_y, tcp_pose_rot6d_col1_z, gripper_position]`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def log_dry_run_summary(
    dataset_dir: Path,
    output_dir: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    state_key: str,
    output_state_key: str,
    parquet_paths: list[Path],
) -> None:
    logger.info("Source dataset: %s", dataset_dir)
    logger.info("Output dataset: %s", output_dir)
    logger.info("Episodes: %d", len(episodes))
    logger.info("Frames: %s", info.get("total_frames", "unknown"))
    logger.info("Parquet files: %d", len(parquet_paths))
    logger.info("State key: %s -> %s", state_key, output_state_key)
    logger.info("Output state shape: [%d]", len(ROT6D_STATE_NAMES))
    logger.info("Output state names: %s", ", ".join(ROT6D_STATE_NAMES))


def validate_input_metadata(info: dict[str, Any], episodes: list[dict[str, Any]]) -> None:
    total_episodes = int(info.get("total_episodes", -1))
    if total_episodes != len(episodes):
        raise ValueError(
            f"info.json total_episodes ({total_episodes}) != episodes.jsonl rows ({len(episodes)})"
        )
    if "features" not in info:
        raise ValueError("info.json is missing features.")


def main() -> None:
    args = parse_args()
    require_pandas()

    dataset_dir = args.dataset_dir.resolve()
    output_dir = resolve_output_dir(dataset_dir, args.output_dir).resolve()

    validate_dataset_dir(dataset_dir)
    info = read_json(dataset_dir / "meta" / "info.json")
    episodes = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    validate_input_metadata(info, episodes)

    state_key = resolve_state_key(info, args.state_key)
    output_state_key = args.output_state_key or state_key
    state_feature = info["features"][state_key]
    state_indices = get_state_indices(state_feature)
    parquet_paths = get_expected_parquet_paths(info, episodes)

    log_dry_run_summary(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        info=info,
        episodes=episodes,
        state_key=state_key,
        output_state_key=output_state_key,
        parquet_paths=parquet_paths,
    )

    if args.dry_run:
        logger.info("Dry-run mode: no files were created.")
        return

    prepare_output_dir(output_dir, args.force)

    state_stats_by_episode: dict[int, dict[str, list[Any]]] = {}
    for idx, (episode, relative_path) in enumerate(zip(episodes, parquet_paths, strict=True), start=1):
        episode_index = int(episode["episode_index"])
        state_stats_by_episode[episode_index] = process_episode_parquet(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            relative_path=relative_path,
            state_key=state_key,
            output_state_key=output_state_key,
            state_indices=state_indices,
        )
        if idx == 1 or idx % 25 == 0 or idx == len(episodes):
            logger.info("Processed parquet episodes: %d/%d", idx, len(episodes))

    total_videos = place_videos(dataset_dir, output_dir, args.video_mode)
    updated_info = update_info(
        info=info,
        state_key=state_key,
        output_state_key=output_state_key,
        total_videos=total_videos,
    )
    updated_episode_stats = build_updated_episode_stats(
        dataset_dir=dataset_dir,
        episodes=episodes,
        state_key=state_key,
        output_state_key=output_state_key,
        state_stats_by_episode=state_stats_by_episode,
    )
    copy_meta_files(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        updated_info=updated_info,
        updated_episode_stats=updated_episode_stats,
    )
    write_readme(output_dir, dataset_dir, state_key, output_state_key)

    logger.info("Done: %s", output_dir)


if __name__ == "__main__":
    main()
