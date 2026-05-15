#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""裁剪 LeRobot v2.1 数据集每个 episode 末尾的若干帧，生成新的数据集目录。

脚本不会修改源数据集，也不会生成 .bak、backup_* 或 *.trim_tail_bak 等自定义备份文件。
输出目录必须不存在；生成后的 data/、videos/、meta/ 路径格式与源数据集 info.json 保持一致。

示例:
  # 预览，不创建输出目录
  python tools/lerobot_dataset_tools/9_trim_tail_frames.py \
      --dataset_dir .cache/huggingface/lerobot/standard/darwin02_0501_2 \
      --output_dir .cache/huggingface/lerobot/standard/darwin02_0501_2_trim_tail \
      --dry_run

  # 删除每个 episode 末尾 20 帧，生成新数据集
  python tools/lerobot_dataset_tools/9_trim_tail_frames.py \
      --dataset_dir .cache/huggingface/lerobot/standard/darwin02_0501_2 \
      --output_dir .cache/huggingface/lerobot/standard/darwin02_0501_2_trim_tail

  # 删除每个 episode 末尾 1 秒；帧数由 info.json 中 fps 推导
  python tools/lerobot_dataset_tools/9_trim_tail_frames.py \
      --dataset_dir .cache/huggingface/lerobot/standard/darwin02_0502 \
      --output_dir .cache/huggingface/lerobot/standard/darwin02_0502_trim_tail \
      --seconds 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EpisodePlan:
    episode_index: int
    old_length: int
    new_length: int
    src_parquet_path: Path
    dst_parquet_path: Path
    video_paths: list[tuple[Path, Path]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="裁剪 LeRobot 数据集每个 episode 末尾的帧，并生成新的数据集目录。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="源 LeRobot 数据集目录。脚本只读取该目录，不会原地修改。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="新数据集输出目录。该目录必须不存在。",
    )
    parser.add_argument(
        "--trim_frames",
        type=int,
        default=20,
        help="每个 episode 末尾删除的帧数。默认 20 帧。",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="按秒数删除末尾数据，帧数按 round(seconds * fps) 计算；会覆盖 --trim_frames。",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印计划，不创建输出目录、不修改文件。",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="只裁剪 parquet 和元数据，不处理 videos/ 下的 mp4。",
    )
    parser.add_argument(
        "--ffmpeg_loglevel",
        default="error",
        help="ffmpeg 日志级别。",
    )
    parser.add_argument(
        "--gop",
        type=int,
        default=2,
        help="ffmpeg -g 参数，即关键帧间隔。默认 2，适合 LeRobot 训练随机读帧。",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def format_dataset_path(
    template: str,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    values = {
        "episode_chunk": episode_chunk(episode_index, chunks_size),
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return Path(template.format(**values))


def get_video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]


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


def recompute_non_visual_stats(
    df: pd.DataFrame,
    features: dict[str, Any],
    previous_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """重算 parquet 中可直接读取的非视觉字段统计，视觉字段沿用旧统计。"""
    stats = dict(previous_stats or {})
    for key, feature in features.items():
        dtype = feature.get("dtype")
        if dtype in {"image", "video", "string"} or key not in df.columns:
            continue

        values = numeric_array_from_series(df[key])
        if len(values) == 0:
            continue
        stats[key] = stats_to_jsonable(values)
    return stats


def ensure_video_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"缺少视频处理工具: {', '.join(missing)}")


def probe_video_frames(video_path: Path) -> int | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if not value or value.upper() == "N/A":
        return None
    return int(value)


def trim_video(
    src_video_path: Path,
    dst_video_path: Path,
    new_length: int,
    fps: int,
    gop: int,
    ffmpeg_loglevel: str,
) -> None:
    dst_video_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"{dst_video_path.stem}.trim_",
        suffix=".mp4",
        dir=dst_video_path.parent,
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        ffmpeg_loglevel,
        "-i",
        str(src_video_path),
        "-map",
        "0:v:0",
        "-an",
        "-frames:v",
        str(new_length),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        actual_frames = probe_video_frames(tmp_path)
        if actual_frames != new_length:
            raise RuntimeError(f"视频裁剪后帧数不匹配: {tmp_path}, 期望 {new_length}, 实际 {actual_frames}")
        tmp_path.replace(dst_video_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def should_skip_auxiliary_file(path: Path) -> bool:
    if ".bak" in path.suffixes:
        return True
    if path.name.endswith(".trim_tail_bak"):
        return True
    if "backup" in path.name:
        return True
    return False


def copy_auxiliary_root_files(dataset_dir: Path, output_dir: Path) -> None:
    for item in dataset_dir.iterdir():
        if not item.is_file() or should_skip_auxiliary_file(item):
            continue
        if item.name in {"README.md"} or item.suffix.lower() == ".json":
            shutil.copy2(item, output_dir / item.name)


def copy_tasks_file(dataset_dir: Path, output_dir: Path) -> None:
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.jsonl 不存在: {tasks_path}")
    dst_path = output_dir / "meta" / "tasks.jsonl"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tasks_path, dst_path)


def build_episode_plans(
    dataset_dir: Path,
    output_dir: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    trim_frames: int,
    include_videos: bool,
) -> list[EpisodePlan]:
    chunks_size = int(info["chunks_size"])
    data_path_template = info["data_path"]
    video_path_template = info.get("video_path")
    video_keys = get_video_keys(info) if include_videos and video_path_template else []

    plans = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        old_length = int(episode["length"])
        if old_length <= trim_frames:
            raise ValueError(f"{dataset_dir} episode {episode_index} 只有 {old_length} 帧，无法删除 {trim_frames} 帧。")

        parquet_rel_path = format_dataset_path(data_path_template, episode_index, chunks_size)
        video_paths = []
        for key in video_keys:
            video_rel_path = format_dataset_path(video_path_template, episode_index, chunks_size, key)
            video_paths.append((dataset_dir / video_rel_path, output_dir / video_rel_path))

        plans.append(
            EpisodePlan(
                episode_index=episode_index,
                old_length=old_length,
                new_length=old_length - trim_frames,
                src_parquet_path=dataset_dir / parquet_rel_path,
                dst_parquet_path=output_dir / parquet_rel_path,
                video_paths=video_paths,
            )
        )
    return plans


def validate_files(plans: list[EpisodePlan], include_videos: bool) -> None:
    missing = []
    for plan in plans:
        if not plan.src_parquet_path.exists():
            missing.append(plan.src_parquet_path)
        if include_videos:
            missing.extend(src_path for src_path, _ in plan.video_paths if not src_path.exists())
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        more = "" if len(missing) <= 20 else f"\n  ... 另有 {len(missing) - 20} 个"
        raise FileNotFoundError(f"发现缺失文件:\n{preview}{more}")


def write_trimmed_parquet(plan: EpisodePlan, global_start_index: int, fps: int) -> pd.DataFrame:
    df = pd.read_parquet(plan.src_parquet_path)
    if len(df) != plan.old_length:
        raise ValueError(
            f"{plan.src_parquet_path} 行数和 episodes.jsonl 不一致: parquet={len(df)}, metadata={plan.old_length}"
        )

    trimmed = df.iloc[: plan.new_length].copy()
    trimmed["frame_index"] = np.arange(plan.new_length, dtype=np.int64)
    trimmed["timestamp"] = (np.arange(plan.new_length, dtype=np.float32) / fps).round(5)
    trimmed["index"] = np.arange(global_start_index, global_start_index + plan.new_length, dtype=np.int64)
    trimmed["episode_index"] = plan.episode_index

    plan.dst_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_parquet(plan.dst_parquet_path, index=False)
    return trimmed


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    trim_frames_arg: int,
    seconds: float | None,
    include_videos: bool,
    dry_run: bool,
    gop: int,
    ffmpeg_loglevel: str,
) -> None:
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"源数据集目录不存在: {dataset_dir}")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，请换一个路径: {output_dir}")
    if not info_path.exists():
        raise FileNotFoundError(f"info.json 不存在: {info_path}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl 不存在: {episodes_path}")

    info = read_json(info_path)
    episodes = read_jsonl(episodes_path)
    stats_rows = read_jsonl(stats_path) if stats_path.exists() else []
    stats_by_ep = {int(row["episode_index"]): row.get("stats", {}) for row in stats_rows}

    fps = int(info["fps"])
    trim_frames = int(round(seconds * fps)) if seconds is not None else trim_frames_arg
    if trim_frames <= 0:
        raise ValueError(f"--trim_frames 必须大于 0，当前为 {trim_frames}")
    if gop <= 0:
        raise ValueError(f"--gop 必须大于 0，当前为 {gop}")

    plans = build_episode_plans(dataset_dir, output_dir, info, episodes, trim_frames, include_videos)
    validate_files(plans, include_videos)

    old_total_frames = int(info["total_frames"])
    new_total_frames = sum(plan.new_length for plan in plans)
    logger.info(
        "源数据集: %s, 输出目录: %s, episodes=%d, fps=%d, 每个 episode 删除 %d 帧，total_frames %d -> %d",
        dataset_dir,
        output_dir,
        len(plans),
        fps,
        trim_frames,
        old_total_frames,
        new_total_frames,
    )

    if dry_run:
        preview = ", ".join(f"ep{plan.episode_index}:{plan.old_length}->{plan.new_length}" for plan in plans[:8])
        suffix = "" if len(plans) <= 8 else f", ... 共 {len(plans)} 个"
        logger.info("dry-run 预览: %s%s", preview, suffix)
        return

    if include_videos:
        ensure_video_tools()

    output_dir.mkdir(parents=True)
    copy_auxiliary_root_files(dataset_dir, output_dir)
    copy_tasks_file(dataset_dir, output_dir)

    new_episode_rows: list[dict[str, Any]] = []
    new_stats_rows: list[dict[str, Any]] = []
    global_start_index = 0

    for i, (episode, plan) in enumerate(zip(episodes, plans, strict=True), start=1):
        if include_videos:
            for src_video_path, dst_video_path in plan.video_paths:
                trim_video(
                    src_video_path=src_video_path,
                    dst_video_path=dst_video_path,
                    new_length=plan.new_length,
                    fps=fps,
                    gop=gop,
                    ffmpeg_loglevel=ffmpeg_loglevel,
                )

        trimmed_df = write_trimmed_parquet(
            plan=plan,
            global_start_index=global_start_index,
            fps=fps,
        )

        updated_episode = dict(episode)
        updated_episode["length"] = plan.new_length
        new_episode_rows.append(updated_episode)

        updated_stats = recompute_non_visual_stats(
            trimmed_df,
            info["features"],
            stats_by_ep.get(plan.episode_index),
        )
        new_stats_rows.append({"episode_index": plan.episode_index, "stats": updated_stats})

        global_start_index += plan.new_length
        if i % 10 == 0 or i == len(plans):
            logger.info("  已处理 %d/%d episodes", i, len(plans))

    output_info = dict(info)
    output_info["total_frames"] = new_total_frames
    output_info["splits"] = {"train": f"0:{len(new_episode_rows)}"}
    output_info["total_episodes"] = len(new_episode_rows)
    output_info["total_chunks"] = max(1, episode_chunk(len(new_episode_rows) - 1, int(info["chunks_size"])) + 1)
    output_info["total_videos"] = len(new_episode_rows) * len(get_video_keys(info))

    write_json(output_dir / "meta" / "info.json", output_info)
    write_jsonl(output_dir / "meta" / "episodes.jsonl", new_episode_rows)
    write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", new_stats_rows)

    logger.info("完成: %s", output_dir)


def main() -> int:
    args = parse_args()
    process_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        trim_frames_arg=args.trim_frames,
        seconds=args.seconds,
        include_videos=not args.no_video,
        dry_run=args.dry_run,
        gop=args.gop,
        ffmpeg_loglevel=args.ffmpeg_loglevel,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
