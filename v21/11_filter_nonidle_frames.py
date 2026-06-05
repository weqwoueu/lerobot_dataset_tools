#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剔除 LeRobot v2.1 数据集中的静止帧，并生成新的数据集目录。

默认不会修改源数据集。输出目录默认是源数据集同级目录的 ``<name>_nonidle``。

示例:
  python tools/lerobot_dataset_tools/11_filter_nonidle_frames.py \
      --dataset_dir /home/standard/workspace/test/kai0/data/standard_Task_A/base/piper_fold_tshirt_red

  python tools/lerobot_dataset_tools/11_filter_nonidle_frames.py \
      --dataset_dir /home/standard/workspace/test/kai0/data/standard_Task_A/base/piper_fold_tshirt_red \
      --dry_run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled in require_pandas()
    pd = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoParams:
    codec_name: str
    codec_tag: str
    pix_fmt: str
    width: int
    height: int
    fps: float
    fps_expr: str
    gop: int
    b_frames: int


@dataclass
class EpisodePlan:
    src_episode_index: int
    dst_episode_index: int | None
    old_length: int
    trimmed_start_frames: int
    trimmed_end_frames: int
    frames_after_time_trim: int
    keep_indices: list[int]
    keep_ranges: list[tuple[int, int]]
    src_parquet_path: Path
    dst_parquet_path: Path | None
    video_paths: list[tuple[Path, Path]]

    @property
    def new_length(self) -> int:
        return len(self.keep_indices)

    @property
    def dropped(self) -> int:
        return self.old_length - self.new_length

    @property
    def drop_ratio(self) -> float:
        if self.old_length == 0:
            return 0.0
        return self.dropped / self.old_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="剔除 LeRobot v2.1 数据集静止帧，并生成新的非静止帧数据集。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="源 LeRobot v2.1 数据集目录。脚本只读取该目录，不会原地修改。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="新数据集输出目录。默认是源目录同级的 <name>_nonidle；目录必须不存在。",
    )
    parser.add_argument(
        "--signal",
        choices=("both", "state", "action"),
        default="both",
        help="用于判断静止的信号。both 表示 concat(observation.state, action)。",
    )
    parser.add_argument("--eps", type=float, default=1e-3, help="静止判断阈值。")
    parser.add_argument("--min_idle_len", type=int, default=7, help="连续静止段至少多少帧才删除。")
    parser.add_argument(
        "--min_nonidle_len",
        type=int,
        default=16,
        help="保留片段至少多少帧；短于该长度的运动碎片会丢弃。",
    )
    parser.add_argument(
        "--filter_last_n",
        type=int,
        default=10,
        help="每个保留片段末尾再裁掉多少帧。",
    )
    parser.add_argument(
        "--trim_start_frames",
        type=int,
        default=0,
        help="静止过滤前，先删除每个 episode 开头的帧数。",
    )
    parser.add_argument(
        "--trim_end_frames",
        type=int,
        default=0,
        help="静止过滤前，先删除每个 episode 结尾的帧数。",
    )
    parser.add_argument(
        "--trim_start_seconds",
        type=float,
        default=None,
        help="按秒数删除每个 episode 开头数据，帧数按 round(seconds * fps) 计算；会覆盖 --trim_start_frames。",
    )
    parser.add_argument(
        "--trim_end_seconds",
        type=float,
        default=None,
        help="按秒数删除每个 episode 结尾数据，帧数按 round(seconds * fps) 计算；会覆盖 --trim_end_frames。",
    )
    parser.add_argument("--dry_run", action="store_true", help="只打印过滤计划和统计，不创建输出目录。")
    parser.add_argument("--no_video", action="store_true", help="只处理 parquet 和 meta，不生成 videos。")
    parser.add_argument("--ffmpeg_loglevel", default="error", help="ffmpeg 日志级别。")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行处理视频的 ffmpeg 任务数。",
    )
    parser.add_argument(
        "--video_codec",
        default="h264_nvenc",
        help="输出视频编码器。默认 h264_nvenc；传 source 表示沿用源视频编码。",
    )
    parser.add_argument(
        "--gop",
        type=int,
        default=2,
        help="输出视频 GOP/关键帧间隔。默认 2，适合训练随机读取 mp4 帧。",
    )
    parser.add_argument(
        "--b_frames",
        type=int,
        default=0,
        help="输出视频 B 帧数量。默认 0，适合训练随机读取 mp4 帧。",
    )
    parser.add_argument(
        "--nvenc_preset",
        default="p4",
        help="使用 h264_nvenc/hevc_nvenc 时的 preset。",
    )
    parser.add_argument(
        "--nvenc_cq",
        type=int,
        default=23,
        help="使用 h264_nvenc/hevc_nvenc 时的 CQ 质量参数。",
    )
    parser.add_argument(
        "--av1_crf",
        type=int,
        default=30,
        help="源视频为 av1 时，重编码使用的 CRF。",
    )
    parser.add_argument(
        "--av1_cpu_used",
        type=int,
        default=8,
        help="源视频为 av1 时，libaom-av1 的 cpu-used 参数。",
    )
    parser.add_argument(
        "--mp4v_qscale",
        type=int,
        default=3,
        help="源视频为 mp4v/mpeg4 时，重编码使用的 qscale。",
    )
    return parser.parse_args()


def require_pandas() -> Any:
    if pd is None:
        raise RuntimeError(
            "缺少 pandas 依赖。请在 lerobot 数据处理环境中运行，或先安装 pandas/pyarrow。"
        )
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


def format_dataset_path(
    template: str,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    values: dict[str, Any] = {
        "episode_chunk": episode_chunk(episode_index, chunks_size),
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return Path(template.format(**values))


def get_video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]


def should_skip_auxiliary_file(path: Path) -> bool:
    if ".bak" in path.suffixes:
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
    src = dataset_dir / "meta" / "tasks.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"tasks.jsonl 不存在: {src}")
    dst = output_dir / "meta" / "tasks.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


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
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"缺少视频处理工具: {', '.join(missing)}")


def ffmpeg_has_encoder(encoder: str) -> bool:
    cmd = ["ffmpeg", "-hide_banner", "-encoders"]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return encoder in result.stdout


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


def probe_keyframe_timestamps(video_path: Path) -> list[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = []
    for line in result.stdout.splitlines():
        value = line.strip().split(",", 1)[0]
        if value and value != "N/A":
            out.append(float(value))
    return out


def parse_fps_expr(value: str) -> float:
    try:
        frac = Fraction(value)
        if frac.denominator != 0:
            return float(frac)
    except Exception:
        pass
    return float(value)


def canonicalize_codec(codec_name: str, codec_tag: str) -> str:
    codec_name = codec_name.strip().lower()
    codec_tag = codec_tag.strip().lower()
    if codec_name == "av1":
        return "av1"
    if codec_name == "mpeg4" and codec_tag == "mp4v":
        return "mp4v"
    if codec_name == "mpeg4":
        return "mp4v"
    return codec_name


def probe_video_params(video_path: Path) -> VideoParams:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,codec_tag_string,pix_fmt,width,height,r_frame_rate,avg_frame_rate,has_b_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"无法读取视频参数: {video_path}")
    stream = streams[0]
    fps_expr = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
    if fps_expr in {"0/0", "N/A"}:
        fps_expr = stream.get("r_frame_rate") or "0/0"
    fps = parse_fps_expr(fps_expr)
    key_ts = probe_keyframe_timestamps(video_path)
    gop = 1
    if len(key_ts) >= 2 and fps > 0:
        gaps = [b - a for a, b in zip(key_ts, key_ts[1:]) if b > a]
        if gaps:
            gop = max(1, int(round(float(np.median(gaps)) * fps)))
    return VideoParams(
        codec_name=canonicalize_codec(stream.get("codec_name", ""), stream.get("codec_tag_string", "")),
        codec_tag=stream.get("codec_tag_string", ""),
        pix_fmt=stream.get("pix_fmt", "yuv420p"),
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=fps,
        fps_expr=fps_expr,
        gop=gop,
        b_frames=int(stream.get("has_b_frames", 0) or 0),
    )


def output_codec_name(params: VideoParams, args: argparse.Namespace) -> str:
    if args.video_codec == "source":
        return params.codec_name
    return str(args.video_codec).strip()


def output_pix_fmt(params: VideoParams, codec: str) -> str:
    if codec in {"h264_nvenc", "hevc_nvenc"}:
        if params.pix_fmt in {"yuv420p", "nv12", "p010le", "yuv444p"}:
            return params.pix_fmt
        return "yuv420p"
    return params.pix_fmt


def output_gop(params: VideoParams, args: argparse.Namespace) -> int:
    return max(1, int(getattr(args, "gop", params.gop)))


def output_b_frames(params: VideoParams, args: argparse.Namespace, gop: int | None = None) -> int:
    gop = max(1, int(gop if gop is not None else output_gop(params, args)))
    b_frames = max(0, int(getattr(args, "b_frames", params.b_frames)))
    if gop <= 1:
        return 0
    return min(b_frames, max(0, gop - 2))


def ffmpeg_codec_args(params: VideoParams, args: argparse.Namespace) -> list[str]:
    codec = output_codec_name(params, args)
    b_frame_args = ["-bf", str(output_b_frames(params, args))]
    if codec in {"h264_nvenc", "hevc_nvenc"}:
        return [
            "-c:v",
            codec,
            "-preset",
            str(args.nvenc_preset),
            "-rc",
            "vbr",
            "-cq",
            str(args.nvenc_cq),
            *b_frame_args,
        ]
    if codec == "av1":
        return [
            "-c:v",
            "libaom-av1",
            "-crf",
            str(args.av1_crf),
            "-b:v",
            "0",
            "-cpu-used",
            str(args.av1_cpu_used),
            "-row-mt",
            "1",
        ]
    if codec == "mp4v":
        return ["-c:v", "mpeg4", "-vtag", "mp4v", "-q:v", str(args.mp4v_qscale), *b_frame_args]
    if codec in {"h264", "avc1"}:
        return ["-c:v", "libx264", *b_frame_args]
    if codec in {"hevc", "h265"}:
        return ["-c:v", "libx265", *b_frame_args]
    logger.warning("未知源视频编码 %s，将尝试使用 ffmpeg 编码器名称直接重编码。", codec)
    return ["-c:v", codec]


def update_video_features_for_output_codec(info: dict[str, Any], output_codec: str) -> dict[str, Any]:
    if output_codec == "source":
        return info
    updated = dict(info)
    features = dict(updated.get("features", {}))
    for key, feature in features.items():
        if feature.get("dtype") != "video":
            continue
        new_feature = dict(feature)
        video_info = dict(new_feature.get("info", {}))
        if output_codec == "h264_nvenc":
            video_info["video.codec"] = "h264"
        elif output_codec == "hevc_nvenc":
            video_info["video.codec"] = "hevc"
        else:
            video_info["video.codec"] = output_codec
        new_feature["info"] = video_info
        features[key] = new_feature
    updated["features"] = features
    return updated


def select_expression_from_ranges(ranges: list[tuple[int, int]]) -> str:
    parts = []
    for start, end in ranges:
        if end <= start:
            continue
        if end == start + 1:
            parts.append(f"eq(n\\,{start})")
        else:
            parts.append(f"between(n\\,{start}\\,{end - 1})")
    if not parts:
        raise ValueError("没有可用于视频过滤的保留片段")
    return "+".join(parts)


def filter_video(
    src_video_path: Path,
    dst_video_path: Path,
    keep_ranges: list[tuple[int, int]],
    expected_frames: int,
    args: argparse.Namespace,
) -> VideoParams:
    params = probe_video_params(src_video_path)
    dst_video_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"{dst_video_path.stem}.nonidle_",
        suffix=".mp4",
        dir=dst_video_path.parent,
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink()

    vf = f"select='{select_expression_from_ranges(keep_ranges)}',setpts=N/FRAME_RATE/TB"
    codec = output_codec_name(params, args)
    gop = output_gop(params, args)
    b_frames = output_b_frames(params, args, gop)
    logger.info(
        "视频编码: %s -> %s, gop=%d, b_frames=%d, %s",
        params.codec_name,
        codec,
        gop,
        b_frames,
        src_video_path,
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        args.ffmpeg_loglevel,
        "-i",
        str(src_video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        vf,
        "-r",
        params.fps_expr,
        *ffmpeg_codec_args(params, args),
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        output_pix_fmt(params, codec),
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        actual_frames = probe_video_frames(tmp_path)
        if actual_frames != expected_frames:
            raise RuntimeError(
                f"视频过滤后帧数不匹配: {src_video_path}, 期望 {expected_frames}, 实际 {actual_frames}"
            )
        tmp_path.replace(dst_video_path)
        return params
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def compute_keep_indices_and_ranges(
    signals: np.ndarray,
    eps: float,
    min_idle_len: int,
    min_nonidle_len: int,
    filter_last_n: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)
    t = int(signals.shape[0])
    if t <= 1 or signals.shape[1] == 0:
        return list(range(t)), [(0, t)] if t > 0 else []

    step_idle = np.all(np.abs(signals[1:] - signals[:-1]) < float(eps), axis=1)
    is_idle_array = np.hstack([np.array([False]), step_idle])
    is_idle_padded = np.concatenate([[False], is_idle_array, [False]])
    is_idle_diff = np.diff(is_idle_padded.astype(int))
    idle_starts = np.where(is_idle_diff == 1)[0]
    idle_ends = np.where(is_idle_diff == -1)[0]

    long_idle = (idle_ends - idle_starts) >= int(min_idle_len)
    idle_starts = idle_starts[long_idle]
    idle_ends = idle_ends[long_idle]

    keep_mask = np.ones(t, dtype=bool)
    for start, end in zip(idle_starts, idle_ends, strict=True):
        keep_mask[int(start) : int(end)] = False

    keep_padded = np.concatenate([[False], keep_mask, [False]])
    keep_diff = np.diff(keep_padded.astype(int))
    keep_starts = np.where(keep_diff == 1)[0]
    keep_ends = np.where(keep_diff == -1)[0]

    ranges: list[tuple[int, int]] = []
    for start, end in zip(keep_starts, keep_ends, strict=True):
        if int(end) - int(start) < int(min_nonidle_len):
            continue
        new_end = int(end) - int(filter_last_n)
        if new_end > int(start):
            ranges.append((int(start), new_end))

    keep_indices: list[int] = []
    for start, end in ranges:
        keep_indices.extend(range(start, end))
    return keep_indices, ranges


def offset_ranges(
    ranges: list[tuple[int, int]],
    offset: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    shifted_ranges = [(start + offset, end + offset) for start, end in ranges]
    keep_indices: list[int] = []
    for start, end in shifted_ranges:
        keep_indices.extend(range(start, end))
    return keep_indices, shifted_ranges


def build_signal(df: pd.DataFrame, signal: str) -> np.ndarray:
    parts = []
    if signal in {"both", "state"}:
        if "observation.state" not in df.columns:
            raise KeyError("parquet 缺少 observation.state 列")
        parts.append(numeric_array_from_series(df["observation.state"]))
    if signal in {"both", "action"}:
        if "action" not in df.columns:
            raise KeyError("parquet 缺少 action 列")
        parts.append(numeric_array_from_series(df["action"]))
    if len(parts) == 1:
        return np.asarray(parts[0], dtype=np.float64)
    return np.concatenate([np.asarray(item, dtype=np.float64) for item in parts], axis=1)


def validate_dataset_dir(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "meta" / "info.json",
        dataset_dir / "meta" / "episodes.jsonl",
        dataset_dir / "data",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少 LeRobot v2.1 数据集文件/目录: " + ", ".join(str(p) for p in missing))


def resolve_trim_frames(args: argparse.Namespace, fps: int) -> tuple[int, int]:
    if args.trim_start_frames < 0 or args.trim_end_frames < 0:
        raise ValueError("--trim_start_frames/--trim_end_frames 不能为负数")
    if args.trim_start_seconds is not None and args.trim_start_seconds < 0:
        raise ValueError("--trim_start_seconds 不能为负数")
    if args.trim_end_seconds is not None and args.trim_end_seconds < 0:
        raise ValueError("--trim_end_seconds 不能为负数")

    trim_start = (
        int(round(float(args.trim_start_seconds) * fps))
        if args.trim_start_seconds is not None
        else int(args.trim_start_frames)
    )
    trim_end = (
        int(round(float(args.trim_end_seconds) * fps))
        if args.trim_end_seconds is not None
        else int(args.trim_end_frames)
    )
    return trim_start, trim_end


def build_plans(
    dataset_dir: Path,
    output_dir: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    args: argparse.Namespace,
    include_videos: bool,
    trim_start_frames: int,
    trim_end_frames: int,
) -> tuple[list[EpisodePlan], list[dict[str, Any]]]:
    chunks_size = int(info["chunks_size"])
    video_keys = get_video_keys(info) if include_videos and info.get("video_path") else []
    plans: list[EpisodePlan] = []
    stats: list[dict[str, Any]] = []
    next_dst_episode_index = 0

    for episode in episodes:
        src_ep = int(episode["episode_index"])
        old_len = int(episode["length"])
        src_parquet = dataset_dir / format_dataset_path(info["data_path"], src_ep, chunks_size)
        if not src_parquet.exists():
            raise FileNotFoundError(f"parquet 不存在: {src_parquet}")

        df = pd.read_parquet(src_parquet)
        if len(df) != old_len:
            raise ValueError(f"{src_parquet} 行数和 episodes.jsonl 不一致: parquet={len(df)}, metadata={old_len}")
        time_start = min(trim_start_frames, old_len)
        time_end = max(time_start, old_len - trim_end_frames)
        frames_after_time_trim = max(0, time_end - time_start)
        keep_indices: list[int] = []
        keep_ranges: list[tuple[int, int]] = []
        if frames_after_time_trim > 0:
            time_trimmed_df = df.iloc[time_start:time_end]
            signal = build_signal(time_trimmed_df, args.signal)
            _local_keep_indices, local_keep_ranges = compute_keep_indices_and_ranges(
                signal,
                eps=args.eps,
                min_idle_len=args.min_idle_len,
                min_nonidle_len=args.min_nonidle_len,
                filter_last_n=args.filter_last_n,
            )
            keep_indices, keep_ranges = offset_ranges(local_keep_ranges, time_start)

        dst_ep: int | None = None
        dst_parquet: Path | None = None
        video_paths: list[tuple[Path, Path]] = []
        if keep_indices:
            dst_ep = next_dst_episode_index
            next_dst_episode_index += 1
            dst_parquet = output_dir / format_dataset_path(info["data_path"], dst_ep, chunks_size)
            for key in video_keys:
                src_video = dataset_dir / format_dataset_path(info["video_path"], src_ep, chunks_size, key)
                dst_video = output_dir / format_dataset_path(info["video_path"], dst_ep, chunks_size, key)
                video_paths.append((src_video, dst_video))

        plan = EpisodePlan(
            src_episode_index=src_ep,
            dst_episode_index=dst_ep,
            old_length=old_len,
            trimmed_start_frames=time_start,
            trimmed_end_frames=old_len - time_end,
            frames_after_time_trim=frames_after_time_trim,
            keep_indices=keep_indices,
            keep_ranges=keep_ranges,
            src_parquet_path=src_parquet,
            dst_parquet_path=dst_parquet,
            video_paths=video_paths,
        )
        plans.append(plan)
        stats.append(episode_stat_row(plan))

    return plans, stats


def episode_stat_row(plan: EpisodePlan) -> dict[str, Any]:
    return {
        "src_episode_index": plan.src_episode_index,
        "dst_episode_index": plan.dst_episode_index,
        "before": plan.old_length,
        "trimmed_start_frames": plan.trimmed_start_frames,
        "trimmed_end_frames": plan.trimmed_end_frames,
        "frames_after_time_trim": plan.frames_after_time_trim,
        "after": plan.new_length,
        "dropped": plan.dropped,
        "drop_ratio": plan.drop_ratio,
        "drop_percent": plan.drop_ratio * 100.0,
        "keep_ranges": [[start, end] for start, end in plan.keep_ranges],
        "keep_range_count": len(plan.keep_ranges),
        "skipped": plan.new_length == 0,
    }


def validate_video_files(plans: list[EpisodePlan], include_videos: bool) -> None:
    if not include_videos:
        return
    missing = []
    for plan in plans:
        if plan.new_length == 0:
            continue
        missing.extend(src for src, _ in plan.video_paths if not src.exists())
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        more = "" if len(missing) <= 20 else f"\n  ... 另有 {len(missing) - 20} 个"
        raise FileNotFoundError(f"发现缺失视频:\n{preview}{more}")


def write_filtered_parquet(plan: EpisodePlan, global_start_index: int, fps: int) -> pd.DataFrame:
    if plan.dst_parquet_path is None or plan.dst_episode_index is None:
        raise ValueError("跳过的 episode 不应写 parquet")
    df = pd.read_parquet(plan.src_parquet_path)
    filtered = df.iloc[plan.keep_indices].copy()
    new_len = len(filtered)
    filtered["frame_index"] = np.arange(new_len, dtype=np.int64)
    filtered["timestamp"] = (np.arange(new_len, dtype=np.float32) / fps).round(5)
    filtered["index"] = np.arange(global_start_index, global_start_index + new_len, dtype=np.int64)
    filtered["episode_index"] = plan.dst_episode_index
    plan.dst_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(plan.dst_parquet_path, index=False)
    return filtered


def filter_videos_parallel(
    kept_plans: list[EpisodePlan],
    args: argparse.Namespace,
) -> None:
    video_tasks = [
        (plan, src_video_path, dst_video_path)
        for plan in kept_plans
        for src_video_path, dst_video_path in plan.video_paths
    ]
    if not video_tasks:
        return
    logger.info(
        "开始并行生成视频: files=%d, workers=%d, video_codec=%s",
        len(video_tasks),
        args.workers,
        args.video_codec,
    )

    def submit_one(task: tuple[EpisodePlan, Path, Path]) -> None:
        plan, src_video_path, dst_video_path = task
        filter_video(
            src_video_path=src_video_path,
            dst_video_path=dst_video_path,
            keep_ranges=plan.keep_ranges,
            expected_frames=plan.new_length,
            args=args,
        )

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(submit_one, task): task for task in video_tasks}
        try:
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                plan, src_video_path, _dst_video_path = task
                try:
                    future.result()
                except Exception as exc:
                    for pending in future_to_task:
                        pending.cancel()
                    raise RuntimeError(
                        f"视频生成失败，停止处理: src_episode={plan.src_episode_index}, video={src_video_path}"
                    ) from exc
                completed += 1
                if completed % 10 == 0 or completed == len(video_tasks):
                    logger.info("  已生成视频 %d/%d", completed, len(video_tasks))
        finally:
            for future in future_to_task:
                if not future.done():
                    future.cancel()


def global_stats(plans: list[EpisodePlan], args: argparse.Namespace) -> dict[str, Any]:
    total_before = sum(plan.old_length for plan in plans)
    total_after = sum(plan.new_length for plan in plans)
    total_dropped = total_before - total_after
    skipped = [plan.src_episode_index for plan in plans if plan.new_length == 0]
    return {
        "parameters": {
            "signal": args.signal,
            "eps": args.eps,
            "min_idle_len": args.min_idle_len,
            "min_nonidle_len": args.min_nonidle_len,
            "filter_last_n": args.filter_last_n,
            "trim_start_frames": getattr(args, "resolved_trim_start_frames", args.trim_start_frames),
            "trim_end_frames": getattr(args, "resolved_trim_end_frames", args.trim_end_frames),
            "trim_start_seconds": args.trim_start_seconds,
            "trim_end_seconds": args.trim_end_seconds,
            "video_codec": getattr(args, "video_codec", None),
            "gop": getattr(args, "gop", None),
            "b_frames": getattr(args, "b_frames", None),
            "workers": getattr(args, "workers", None),
            "nvenc_preset": getattr(args, "nvenc_preset", None),
            "nvenc_cq": getattr(args, "nvenc_cq", None),
        },
        "summary": {
            "episodes_before": len(plans),
            "episodes_after": len(plans) - len(skipped),
            "skipped_empty_episodes": len(skipped),
            "skipped_src_episode_indices": skipped,
            "frames_before": total_before,
            "frames_after": total_after,
            "frames_dropped": total_dropped,
            "drop_ratio": (total_dropped / total_before) if total_before else 0.0,
            "drop_percent": ((total_dropped / total_before) * 100.0) if total_before else 0.0,
        },
        "episodes": [episode_stat_row(plan) for plan in plans],
    }


def log_summary(plans: list[EpisodePlan], output_dir: Path) -> None:
    stats = global_stats(plans, argparse.Namespace(
        signal="",
        eps=0,
        min_idle_len=0,
        min_nonidle_len=0,
        filter_last_n=0,
        trim_start_frames=0,
        trim_end_frames=0,
        trim_start_seconds=None,
        trim_end_seconds=None,
    ))["summary"]
    logger.info(
        "total_frames: %d -> %d",
        stats["frames_before"],
        stats["frames_after"],
    )
    logger.info(
        "dropped: %d (%.2f%%)",
        stats["frames_dropped"],
        stats["drop_percent"],
    )
    logger.info(
        "episodes: %d -> %d",
        stats["episodes_before"],
        stats["episodes_after"],
    )
    logger.info("skipped_empty_episodes: %d", stats["skipped_empty_episodes"])
    logger.info("输出目录: %s", output_dir)


def process_dataset(args: argparse.Namespace) -> None:
    require_pandas()

    dataset_dir = args.dataset_dir.resolve()
    output_dir = (args.output_dir or dataset_dir.with_name(f"{dataset_dir.name}_nonidle")).resolve()
    include_videos = not args.no_video

    validate_dataset_dir(dataset_dir)
    if include_videos and not (dataset_dir / "videos").is_dir():
        raise FileNotFoundError(f"videos/ 目录不存在: {dataset_dir / 'videos'}")
    if output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"输出目录已存在，请换一个路径: {output_dir}")
    if args.eps <= 0:
        raise ValueError("--eps 必须大于 0")
    if args.min_idle_len <= 0 or args.min_nonidle_len <= 0 or args.filter_last_n < 0:
        raise ValueError("--min_idle_len/--min_nonidle_len 必须大于 0，--filter_last_n 不能为负数")
    if args.workers <= 0:
        raise ValueError("--workers 必须大于 0")
    if args.gop <= 0:
        raise ValueError("--gop 必须大于 0")
    if args.b_frames < 0:
        raise ValueError("--b_frames 必须大于等于 0")

    info = read_json(dataset_dir / "meta" / "info.json")
    fps = int(info["fps"])
    trim_start_frames, trim_end_frames = resolve_trim_frames(args, fps)
    args.resolved_trim_start_frames = trim_start_frames
    args.resolved_trim_end_frames = trim_end_frames
    episodes = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    episodes_by_index = {int(row["episode_index"]): row for row in episodes}
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    old_stats_rows = read_jsonl(stats_path) if stats_path.exists() else []
    old_stats_by_ep = {int(row["episode_index"]): row.get("stats", {}) for row in old_stats_rows}

    plans, _ = build_plans(
        dataset_dir,
        output_dir,
        info,
        episodes,
        args,
        include_videos,
        trim_start_frames,
        trim_end_frames,
    )
    validate_video_files(plans, include_videos)
    log_summary(plans, output_dir)

    if args.dry_run:
        for plan in plans[:20]:
            logger.info(
                "dry-run ep%d -> %s: %d -> time_trim %d -> nonidle %d, dropped=%d (%.2f%%), ranges=%d",
                plan.src_episode_index,
                "skip" if plan.dst_episode_index is None else f"ep{plan.dst_episode_index}",
                plan.old_length,
                plan.frames_after_time_trim,
                plan.new_length,
                plan.dropped,
                plan.drop_ratio * 100.0,
                len(plan.keep_ranges),
            )
        if len(plans) > 20:
            logger.info("dry-run 另有 %d 个 episode 未逐条列出", len(plans) - 20)
        return

    if include_videos:
        ensure_video_tools()
        if args.video_codec != "source" and not ffmpeg_has_encoder(args.video_codec):
            raise RuntimeError(f"当前 ffmpeg 不支持编码器: {args.video_codec}")

    chunks_size = int(info["chunks_size"])
    output_dir.mkdir(parents=True)
    copy_auxiliary_root_files(dataset_dir, output_dir)
    copy_tasks_file(dataset_dir, output_dir)

    new_episode_rows: list[dict[str, Any]] = []
    new_stats_rows: list[dict[str, Any]] = []
    global_start_index = 0

    kept_plans = [plan for plan in plans if plan.new_length > 0]
    for i, plan in enumerate(kept_plans, start=1):
        if plan.dst_episode_index is None:
            raise ValueError("保留 episode 缺少目标 episode_index")

        filtered_df = write_filtered_parquet(plan, global_start_index, fps)

        old_episode = episodes_by_index[plan.src_episode_index]
        new_episode = dict(old_episode)
        new_episode["episode_index"] = plan.dst_episode_index
        new_episode["length"] = plan.new_length
        new_episode_rows.append(new_episode)

        old_stats = old_stats_by_ep.get(plan.src_episode_index)
        new_stats = recompute_non_visual_stats(filtered_df, info["features"], old_stats)
        new_stats_rows.append({"episode_index": plan.dst_episode_index, "stats": new_stats})

        global_start_index += plan.new_length
        if i % 10 == 0 or i == len(kept_plans):
            logger.info("  已写入 parquet %d/%d kept episodes", i, len(kept_plans))

    if include_videos:
        filter_videos_parallel(kept_plans, args)

    output_info = update_video_features_for_output_codec(dict(info), args.video_codec)
    output_info["total_frames"] = global_start_index
    output_info["splits"] = {"train": f"0:{len(new_episode_rows)}"}
    output_info["total_episodes"] = len(new_episode_rows)
    output_info["total_chunks"] = (
        max(1, episode_chunk(len(new_episode_rows) - 1, chunks_size) + 1) if new_episode_rows else 0
    )
    output_info["total_videos"] = len(new_episode_rows) * len(get_video_keys(info))

    write_json(output_dir / "meta" / "info.json", output_info)
    write_jsonl(output_dir / "meta" / "episodes.jsonl", new_episode_rows)
    write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", new_stats_rows)
    write_json(output_dir / "meta" / "nonidle_filter_stats.json", global_stats(plans, args))
    logger.info("完成: %s", output_dir)


def main() -> int:
    args = parse_args()
    process_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
