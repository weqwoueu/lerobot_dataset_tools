#!/usr/bin/env python3
"""
extract_lerobot.py

Extract (downsample) frames from a LeRobot dataset by keeping every Nth frame.
This accelerates the actions in the dataset - for example, with extraction_factor=2,
a 60-frame episode becomes a 30-frame episode.

Usage:
    python extract_lerobot.py --src_path /path/to/source --tgt_path /path/to/target \\
                              --repo_id extracted_dataset --extraction_factor 2 --num-workers 4
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# --- lerobot imports (must be available in env) ---
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import (
        load_episodes,
        load_episodes_stats,
        load_info,
        load_tasks,
    )
except Exception as e:
    raise RuntimeError("lerobot package import failed. Activate environment where lerobot is installed.") from e

# Try to import merge function
try:
    from merge_lerobot import merge_repos
    MERGE_AVAILABLE = True
except ImportError:
    MERGE_AVAILABLE = False
    print("Warning: merge_lerobot module not available. Merge functionality will be disabled.")


NVENC_MIN_SIDE_FOR_SCRIPT = 256


@dataclass(frozen=True)
class VideoEncodeConfig:
    """FFmpeg encoding options aligned with 11_filter_nonidle_frames.py."""

    video_codec: str = "h264_nvenc"
    gop: int = 2
    b_frames: int = 0
    nvenc_preset: str = "p4"
    nvenc_cq: int = 23
    av1_crf: int = 30
    av1_cpu_used: int = 8
    mp4v_qscale: int = 3
    ffmpeg_loglevel: str = "error"


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


@dataclass(frozen=True)
class ResolvedVideoEncoding:
    codec: str
    encoder: str
    codec_args: Tuple[str, ...]
    pix_fmt: str
    gop: int
    b_frames: int
    used_nvenc_fallback: bool


@dataclass(frozen=True)
class VideoScaleTask:
    label: str
    src_path: Path
    tgt_path: Path
    extraction_factor: int
    expected_source_frames: int
    expected_frames: int
    params: VideoParams
    encoding: ResolvedVideoEncoding


def find_parquet_by_episode(src_root: Path, ep_idx: int) -> Path | None:
    """Try to locate parquet for given episode index under src_root."""
    basename = f"episode_{ep_idx:06d}.parquet"
    candidates = list(src_root.rglob(basename))
    return candidates[0] if candidates else None


def find_video_by_episode_and_key(src_root: Path, ep_idx: int, vid_key: str) -> Path | None:
    """Try to locate mp4 for given episode and video key under src_root."""
    basename = f"episode_{ep_idx:06d}.mp4"
    candidates = list(src_root.rglob(basename))
    for c in candidates:
        if vid_key in "/".join(c.parts):
            return c
    return candidates[0] if candidates else None


def write_parquet_like_source(df: pd.DataFrame, src_parquet_path: Path, tgt_parquet_path: Path) -> None:
    """Write parquet with the same Arrow field types/order as the source episode."""
    source_schema = pq.read_schema(str(src_parquet_path))
    source_names = source_schema.names

    ordered_columns = [name for name in source_names if name in df.columns]
    extra_columns = [name for name in df.columns if name not in source_names]
    output_df = df[ordered_columns + extra_columns]

    if not extra_columns and len(ordered_columns) == len(source_names):
        target_schema = source_schema
    else:
        inferred_schema = pa.Table.from_pandas(output_df, preserve_index=False).schema
        target_schema = pa.schema([
            source_schema.field(name) if name in source_names else inferred_schema.field(name)
            for name in output_df.columns
        ])

    table = pa.Table.from_pandas(output_df, schema=target_schema, preserve_index=False)
    pq.write_table(table, str(tgt_parquet_path))


def validate_video_encode_config(config: VideoEncodeConfig, num_workers: int) -> None:
    allowed_codecs = {
        "source",
        "h264_nvenc",
        "hevc_nvenc",
        "h264",
        "hevc",
        "h265",
        "av1",
        "mp4v",
    }
    if config.video_codec not in allowed_codecs:
        raise ValueError(f"--video-codec 不支持: {config.video_codec!r}，可选 {sorted(allowed_codecs)}")
    if config.gop <= 0:
        raise ValueError("--gop 必须大于 0")
    if config.b_frames < 0:
        raise ValueError("--b-frames 必须大于等于 0")
    if num_workers <= 0:
        raise ValueError("--num-workers 必须大于 0")


def ensure_video_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"缺少视频处理工具: {', '.join(missing)}")


def ffmpeg_has_encoder(encoder: str) -> bool:
    cmd = ["ffmpeg", "-hide_banner", "-encoders"]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return encoder in result.stdout


def _parse_optional_int(value: str) -> Optional[int]:
    stripped = value.strip()
    if not stripped or stripped.upper() == "N/A":
        return None
    return int(stripped)


def probe_video_metadata_frames(video_path: Path) -> Optional[int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return _parse_optional_int(result.stdout)


def probe_video_frames(video_path: Path, exact: bool = False) -> Optional[int]:
    if not exact:
        metadata_frames = probe_video_metadata_frames(video_path)
        if metadata_frames is not None:
            return metadata_frames

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
    return _parse_optional_int(result.stdout)


def probe_keyframe_timestamps(video_path: Path) -> List[float]:
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
    timestamps = []
    for line in result.stdout.splitlines():
        value = line.strip().split(",", 1)[0]
        if value and value != "N/A":
            timestamps.append(float(value))
    return timestamps


def parse_fps_expr(value: str) -> float:
    try:
        fraction = Fraction(value)
        if fraction.denominator != 0:
            return float(fraction)
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


def probe_video_params(video_path: Path, probe_gop: bool = False) -> VideoParams:
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
    if fps <= 0:
        raise ValueError(f"视频帧率无效: {video_path}, fps={fps_expr}")
    gop = 1
    keyframe_timestamps = probe_keyframe_timestamps(video_path) if probe_gop else []
    if len(keyframe_timestamps) >= 2:
        gaps = [
            end - start
            for start, end in zip(keyframe_timestamps, keyframe_timestamps[1:], strict=False)
            if end > start
        ]
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


def output_codec_name(params: VideoParams, config: VideoEncodeConfig) -> Tuple[str, bool]:
    requested_codec = params.codec_name if config.video_codec == "source" else config.video_codec.strip()
    fallback_map = {"h264_nvenc": "h264", "hevc_nvenc": "hevc"}
    fallback_codec = fallback_map.get(requested_codec)
    if fallback_codec and (
        params.width < NVENC_MIN_SIDE_FOR_SCRIPT
        or params.height < NVENC_MIN_SIDE_FOR_SCRIPT
    ):
        return fallback_codec, True
    return requested_codec, False


def output_pix_fmt(params: VideoParams, codec: str) -> str:
    if codec in {"h264_nvenc", "hevc_nvenc"}:
        if params.pix_fmt in {"yuv420p", "nv12", "p010le", "yuv444p"}:
            return params.pix_fmt
        return "yuv420p"
    return params.pix_fmt


def output_b_frames(config: VideoEncodeConfig) -> int:
    if config.gop <= 1:
        return 0
    return min(config.b_frames, max(0, config.gop - 2))


def resolve_video_encoding(params: VideoParams, config: VideoEncodeConfig) -> ResolvedVideoEncoding:
    codec, used_fallback = output_codec_name(params, config)
    b_frames = output_b_frames(config)
    b_frame_args = ["-bf", str(b_frames)]
    if codec in {"h264_nvenc", "hevc_nvenc"}:
        encoder = codec
        codec_args = [
            "-c:v",
            encoder,
            "-preset",
            config.nvenc_preset,
            "-rc",
            "vbr",
            "-cq",
            str(config.nvenc_cq),
            *b_frame_args,
        ]
    elif codec == "av1":
        encoder = "libaom-av1"
        codec_args = [
            "-c:v",
            encoder,
            "-crf",
            str(config.av1_crf),
            "-b:v",
            "0",
            "-cpu-used",
            str(config.av1_cpu_used),
            "-row-mt",
            "1",
        ]
    elif codec == "mp4v":
        encoder = "mpeg4"
        codec_args = [
            "-c:v",
            encoder,
            "-vtag",
            "mp4v",
            "-q:v",
            str(config.mp4v_qscale),
            *b_frame_args,
        ]
    elif codec in {"h264", "avc1"}:
        encoder = "libx264"
        codec_args = ["-c:v", encoder, *b_frame_args]
    elif codec in {"hevc", "h265"}:
        encoder = "libx265"
        codec_args = ["-c:v", encoder, *b_frame_args]
    else:
        encoder = codec
        codec_args = ["-c:v", encoder]
    return ResolvedVideoEncoding(
        codec=codec,
        encoder=encoder,
        codec_args=tuple(codec_args),
        pix_fmt=output_pix_fmt(params, codec),
        gop=config.gop,
        b_frames=b_frames,
        used_nvenc_fallback=used_fallback,
    )


def expected_extracted_length(source_length: int, extraction_factor: int) -> int:
    return (source_length + extraction_factor - 1) // extraction_factor


def build_video_scale_task(task: Dict[str, Any], config: VideoEncodeConfig) -> VideoScaleTask:
    src_path = Path(task["src_path"])
    expected_source_frames = int(task["source_length"])
    source_frames = probe_video_frames(src_path)
    if source_frames != expected_source_frames:
        raise ValueError(
            f"源视频帧数与 episode length 不一致: {src_path}, "
            f"episode={expected_source_frames}, video={source_frames}"
        )
    params = probe_video_params(src_path)
    return VideoScaleTask(
        label=str(task["label"]),
        src_path=src_path,
        tgt_path=Path(task["tgt_path"]),
        extraction_factor=int(task["extraction_factor"]),
        expected_source_frames=expected_source_frames,
        expected_frames=int(task["target_length"]),
        params=params,
        encoding=resolve_video_encoding(params, config),
    )


def preflight_video_scale_tasks(
    tasks: List[Dict[str, Any]],
    config: VideoEncodeConfig,
    num_workers: int,
    desc: str,
) -> List[VideoScaleTask]:
    if not tasks:
        return []
    ensure_video_tools()
    max_workers = max(1, min(num_workers, len(tasks)))
    print(f"[*] {desc} 预检视频: files={len(tasks)}, workers={max_workers}")
    resolved: List[Optional[VideoScaleTask]] = [None] * len(tasks)
    if max_workers == 1:
        for index, task in enumerate(tqdm(tasks, desc=f"Probing {desc}")):
            resolved[index] = build_video_scale_task(task, config)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(build_video_scale_task, task, config): index
                for index, task in enumerate(tasks)
            }
            try:
                for future in tqdm(
                    as_completed(future_to_index),
                    total=len(future_to_index),
                    desc=f"Probing {desc}",
                ):
                    resolved[future_to_index[future]] = future.result()
            except Exception:
                for pending in future_to_index:
                    pending.cancel()
                raise
    video_tasks = [task for task in resolved if task is not None]
    required_encoders = {task.encoding.encoder for task in video_tasks}
    missing_encoders = sorted(
        encoder for encoder in required_encoders if not ffmpeg_has_encoder(encoder)
    )
    if missing_encoders:
        raise RuntimeError(f"当前 ffmpeg 不支持编码器: {', '.join(missing_encoders)}")
    return video_tasks


def ffmpeg_select_filter(extraction_factor: int) -> str:
    if extraction_factor <= 1:
        return "setpts=N/FRAME_RATE/TB"
    return f"select='not(mod(n\\,{extraction_factor}))',setpts=N/FRAME_RATE/TB"


def build_ffmpeg_scale_command(
    task: VideoScaleTask,
    tmp_path: Path,
    config: VideoEncodeConfig,
) -> List[str]:
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        config.ffmpeg_loglevel,
        "-i",
        str(task.src_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        ffmpeg_select_filter(task.extraction_factor),
        "-r",
        task.params.fps_expr,
        *task.encoding.codec_args,
        "-g",
        str(task.encoding.gop),
        "-keyint_min",
        str(task.encoding.gop),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        task.encoding.pix_fmt,
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]


def run_video_scale_task(task: VideoScaleTask, config: VideoEncodeConfig) -> None:
    task.tgt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"{task.tgt_path.stem}.time_scaling_",
        suffix=".mp4",
        dir=task.tgt_path.parent,
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink()
    cmd = build_ffmpeg_scale_command(task, tmp_path, config)
    try:
        subprocess.run(cmd, check=True)
        actual_frames = probe_video_frames(tmp_path, exact=True)
        if actual_frames != task.expected_frames:
            raise RuntimeError(
                f"视频时间缩放后帧数不匹配: {task.src_path}, "
                f"期望 {task.expected_frames}, 实际 {actual_frames}"
            )
        tmp_path.replace(task.tgt_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def process_video_scale_tasks(
    tasks: List[Dict[str, Any]],
    num_workers: int,
    config: VideoEncodeConfig,
    desc: str = "Extracting videos",
) -> List[str]:
    video_tasks = preflight_video_scale_tasks(tasks, config, num_workers, desc)
    if not video_tasks:
        return []
    max_workers = max(1, min(num_workers, len(video_tasks)))
    print(
        f"[*] {desc}: files={len(video_tasks)}, workers={max_workers}, "
        f"video_codec={config.video_codec}"
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_video_scale_task, task, config): task
            for task in video_tasks
        }
        try:
            for future in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc=desc,
            ):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as exc:
                    for pending in future_to_task:
                        pending.cancel()
                    raise RuntimeError(
                        f"视频生成失败，停止处理: {task.label}, video={task.src_path}"
                    ) from exc
        finally:
            for future in future_to_task:
                future.cancel()
    print(f"[*] {desc} complete: {len(video_tasks)} succeeded, 0 failed")
    return []


def _copy_video_task(task: Dict[str, Any]) -> Tuple[str, bool, str]:
    label = task["label"]
    try:
        src_path = Path(task["src_path"])
        tgt_path = Path(task["tgt_path"])
        expected_frames = int(task.get("source_length", 0))
        if expected_frames > 0:
            source_frames = probe_video_frames(src_path)
            if source_frames != expected_frames:
                raise ValueError(
                    f"源视频帧数与 episode length 不一致: {src_path}, "
                    f"episode={expected_frames}, video={source_frames}"
                )
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"{tgt_path.stem}.copy_",
            suffix=tgt_path.suffix,
            dir=tgt_path.parent,
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            shutil.copy2(str(src_path), str(tmp_path))
            tmp_path.replace(tgt_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return label, True, "Copied video"
    except Exception as e:
        return label, False, f"Failed to copy {label}: {e}"


def process_video_copy_tasks(tasks: List[Dict[str, Any]], num_workers: int) -> List[str]:
    if not tasks:
        return []
    ensure_video_tools()
    max_workers = max(1, min(num_workers, len(tasks)))
    print(f"[*] Copying videos: files={len(tasks)}, workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {
            executor.submit(_copy_video_task, task): task["label"]
            for task in tasks
        }
        try:
            for future in tqdm(
                as_completed(future_to_label),
                total=len(future_to_label),
                desc="Copying videos",
            ):
                label = future_to_label[future]
                try:
                    _label, success, message = future.result()
                except Exception as exc:
                    for pending in future_to_label:
                        pending.cancel()
                    raise RuntimeError(f"视频复制失败，停止处理: {label}") from exc
                if not success:
                    for pending in future_to_label:
                        pending.cancel()
                    raise RuntimeError(message)
        finally:
            for future in future_to_label:
                future.cancel()
    print(f"[*] Copying videos complete: {len(tasks)} succeeded, 0 failed")
    return []


def output_codec_metadata_name(config: VideoEncodeConfig) -> Optional[str]:
    if config.video_codec == "source":
        return None
    return {
        "h264_nvenc": "h264",
        "hevc_nvenc": "hevc",
        "h265": "hevc",
    }.get(config.video_codec, config.video_codec)


def output_features_for_video_config(
    features: Dict[str, Any],
    config: VideoEncodeConfig,
) -> Dict[str, Any]:
    output_features = json.loads(json.dumps(features))
    codec_name = output_codec_metadata_name(config)
    if codec_name is None:
        return output_features
    for _key, feature in output_features.items():
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            continue
        video_info = dict(feature.get("info", {}))
        video_info["video.codec"] = codec_name
        feature["info"] = video_info
    return output_features


def update_info_json_video_codec(dataset_root: Path, config: VideoEncodeConfig) -> None:
    codec_name = output_codec_metadata_name(config)
    if codec_name is None:
        return
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    features = info.get("features", {})
    if isinstance(features, dict):
        info["features"] = output_features_for_video_config(features, config)
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
        f.write("\n")


def print_time_scaling_plan(
    *,
    src_root: Path,
    tgt_root: Path,
    repo_id: str,
    fps: float,
    robot_type: str,
    total_episodes: int,
    total_frames: int,
    extraction_factor: int,
    num_workers: int,
    video_keys: Sequence[str],
    video_config: VideoEncodeConfig,
    split_ratio: Optional[float] = None,
    extract_count: Optional[int] = None,
    keep_count: Optional[int] = None,
) -> None:
    print("=" * 72)
    print("时间缩放转换方案（核心逻辑：每 N 帧保留第 1 帧，timestamp 重新按 fps 连续生成）")
    print("=" * 72)
    print(f"源数据集: {src_root}")
    print(f"输出数据集: {tgt_root}")
    print(f"repo_id: {repo_id}")
    print(f"FPS: {fps}")
    print(f"Robot type: {robot_type}")
    print(f"源 episodes/frames: {total_episodes}/{total_frames}")
    if split_ratio is None:
        print("模式: 全量时间缩放")
    else:
        print(f"模式: split 时间增强，split_ratio={split_ratio}")
        print(f"  时间缩放 episodes: {extract_count}")
        print(f"  保留原速 episodes: {keep_count}")
    print(f"extraction_factor: {extraction_factor}")
    print("Parquet: 保留行 0, N, 2N...；episode_index/frame_index/index/timestamp 重新连续编号")
    print(f"视频 key: {list(video_keys)}")
    print("视频: FFmpeg select,setpts 重新编码；不保留音频；成功后原子替换")
    print(f"视频 codec: {video_config.video_codec}")
    print(f"GOP/B frames: {video_config.gop}/{output_b_frames(video_config)}")
    print("视频校验: 源视频帧数等于源 episode length，输出视频帧数等于抽取后 length")
    print(f"并行 workers: {num_workers}")
    print("=" * 72)
    print()


def extract_dataset(
    src_path: str,
    tgt_path: str,
    repo_id: str,
    extraction_factor: int = 2,
    num_workers: int = 4,
    force: bool = False,
    merge_src_paths: List[str] = None,
    merge_tgt_path: str = None,
    merge_repo_id: str = None,
    merge_force: bool = False,
    video_config: Optional[VideoEncodeConfig] = None,
):
    """
    Extract frames from a LeRobot dataset by keeping every Nth frame.

    Args:
        src_path: Path to source LeRobot dataset
        tgt_path: Path to target (extracted) dataset
        repo_id: Repository ID for the new dataset
        extraction_factor: Keep every Nth frame (e.g., 2 means keep frames 0, 2, 4, ...)
        num_workers: Number of parallel worker processes for video processing
        force: Force extraction even if target exists
        merge_src_paths: Optional list of additional source dataset paths to merge with extracted dataset
        merge_tgt_path: Optional path for merged dataset (if None and merge_src_paths provided, uses tgt_path)
        merge_repo_id: Optional repo_id for merged dataset (if None and merge_src_paths provided, uses repo_id)
        merge_force: Force merge even if conflicts exist
    """
    if extraction_factor < 1:
        raise ValueError(f"extraction_factor must be >= 1, got {extraction_factor}")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {num_workers}")
    video_config = video_config or VideoEncodeConfig()
    validate_video_encode_config(video_config, num_workers)

    src_root = Path(src_path).expanduser().resolve()
    if not src_root.exists():
        raise RuntimeError(f"Source path does not exist: {src_path}")

    tgt_root = Path(tgt_path).expanduser().resolve()
    if tgt_root.exists() and any(tgt_root.iterdir()) and not force:
        raise RuntimeError(f"Target {tgt_root} exists and is not empty. Use --force or remove it first.")

    # Load source dataset info
    print(f"[*] Loading source dataset from {src_root}")
    try:
        src_info = load_info(src_root)
        src_episodes = load_episodes(src_root)
        src_tasks_dict, _ = load_tasks(src_root)
    except Exception as e:
        raise RuntimeError(f"Cannot load source dataset: {e}") from e

    try:
        src_episodes_stats = load_episodes_stats(src_root)
    except Exception:
        src_episodes_stats = {}

    # Extract parameters from source
    fps = src_info.get("fps", 30)
    robot_type = src_info.get("robot_type", "unknown")
    features = src_info.get("features", {})
    output_features = output_features_for_video_config(features, video_config)
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    total_episodes = int(src_info.get("total_episodes", len(src_episodes)))
    total_frames = int(src_info.get("total_frames", 0))

    print_time_scaling_plan(
        src_root=src_root,
        tgt_root=tgt_root,
        repo_id=repo_id,
        fps=float(fps),
        robot_type=robot_type,
        total_episodes=total_episodes,
        total_frames=total_frames,
        extraction_factor=extraction_factor,
        num_workers=num_workers,
        video_keys=video_keys,
        video_config=video_config,
    )

    print("[*] Source dataset info:")
    print(f"    - FPS: {fps}")
    print(f"    - Robot type: {robot_type}")
    print(f"    - Total episodes: {src_info.get('total_episodes', 0)}")
    print(f"    - Total frames: {src_info.get('total_frames', 0)}")
    print(f"    - Extraction factor: {extraction_factor}")
    print(f"    - Video workers: {num_workers}")
    print(f"    - Video codec: {video_config.video_codec}")

    # Create target dataset structure
    print(f"[*] Creating target dataset at {tgt_root}")
    ds_target = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        root=str(tgt_root),
        robot_type=robot_type,
        features=output_features
    )
    meta_target = ds_target.meta

    # Add tasks from source
    for _k, task in sorted(src_tasks_dict.items(), key=lambda kv: int(kv[0])):
        meta_target.add_task(task)
    print(f"[*] Added {len(src_tasks_dict)} tasks to target dataset")

    print(f"[*] Video keys to process: {video_keys}")

    warnings: List[str] = []
    video_tasks: List[Dict[str, Any]] = []

    # Process each episode
    items_sorted = sorted(src_episodes.items(), key=lambda kv: int(kv[0]))
    for src_ep_idx_str, ep in tqdm(items_sorted, desc="Extracting episodes"):
        src_ep_idx = int(src_ep_idx_str)
        src_ep_length = int(ep.get("length", 0))
        ep_tasks = ep.get("tasks", [])

        # Calculate new episode length after extraction
        new_ep_length = (src_ep_length + extraction_factor - 1) // extraction_factor
        if new_ep_length == 0:
            warnings.append(f"Episode {src_ep_idx} too short ({src_ep_length} frames), skipping")
            continue

        # Find source parquet file
        src_chunksize = int(src_info.get("chunks_size", 1000))
        src_chunk_idx = src_ep_idx // src_chunksize

        try:
            src_parquet_rel = src_info["data_path"].format(
                episode_chunk=src_chunk_idx,
                episode_index=src_ep_idx
            )
            src_parquet_path = (src_root / src_parquet_rel).resolve()
            if not src_parquet_path.is_file():
                alt = find_parquet_by_episode(src_root, src_ep_idx)
                if alt:
                    src_parquet_path = alt
                else:
                    raise FileNotFoundError
        except Exception:
            alt = find_parquet_by_episode(src_root, src_ep_idx)
            if alt:
                src_parquet_path = alt
            else:
                warnings.append(f"Parquet for episode {src_ep_idx} not found, skipping")
                continue

        # New episode index in target
        new_ep_idx = int(meta_target.info["total_episodes"])

        # Target paths
        tgt_chunk_idx = meta_target.get_episode_chunk(new_ep_idx)
        tgt_parquet_rel = meta_target.data_path.format(
            episode_chunk=tgt_chunk_idx,
            episode_index=new_ep_idx
        )
        tgt_parquet_path = meta_target.root / tgt_parquet_rel

        # --- EXTRACT & PATCH PARQUET: keep every Nth frame ---
        start_global_index = int(meta_target.info.get("total_frames", 0))
        tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_parquet(str(src_parquet_path))
            total_frames = len(df)

            # Extract every Nth frame (indices 0, N, 2N, 3N, ...)
            extracted_indices = list(range(0, total_frames, extraction_factor))
            df_extracted = df.iloc[extracted_indices].copy()

            # Reset pandas index immediately to avoid alignment issues
            df_extracted.reset_index(drop=True, inplace=True)

            n_rows = len(df_extracted)

            # Helper function to handle episode_index column format
            def make_episode_index_col(series, scalar_val, n):
                if len(series) > 0 and series.dtype == object and isinstance(series.iloc[0], (list, tuple, np.ndarray)):
                    return pd.Series([[int(scalar_val)]] * n, index=range(n))
                else:
                    return pd.Series([int(scalar_val)] * n, index=range(n))

            # Update episode_index
            df_extracted["episode_index"] = make_episode_index_col(df_extracted["episode_index"], new_ep_idx, n_rows)

            # Update frame_index to be sequential within the episode [0, 1, 2, 3, ...]
            if "frame_index" in df_extracted.columns:
                new_frame_indices = list(range(n_rows))
                if df_extracted["frame_index"].dtype == object and n_rows > 0:
                    first_val = df_extracted["frame_index"].iloc[0]
                    if isinstance(first_val, (list, tuple, np.ndarray)):
                        df_extracted["frame_index"] = [[int(x)] for x in new_frame_indices]
                    else:
                        df_extracted["frame_index"] = new_frame_indices
                else:
                    df_extracted["frame_index"] = new_frame_indices

            # Update timestamp to be evenly spaced according to FPS
            if "timestamp" in df_extracted.columns:
                frame_duration = 1.0 / fps
                new_timestamps = (np.arange(n_rows, dtype=np.float32) * np.float32(frame_duration)).astype(np.float32)
                if df_extracted["timestamp"].dtype == object and n_rows > 0:
                    first_val = df_extracted["timestamp"].iloc[0]
                    if isinstance(first_val, (list, tuple, np.ndarray)):
                        df_extracted["timestamp"] = [[float(x)] for x in new_timestamps]
                    else:
                        df_extracted["timestamp"] = new_timestamps
                else:
                    df_extracted["timestamp"] = new_timestamps

            # Update global index
            new_global_indices = list(range(start_global_index, start_global_index + n_rows))
            if "index" in df_extracted.columns:
                if df_extracted["index"].dtype == object and n_rows > 0:
                    first_val = df_extracted["index"].iloc[0]
                    if isinstance(first_val, (list, tuple, np.ndarray)):
                        df_extracted["index"] = [[int(x)] for x in new_global_indices]
                    else:
                        df_extracted["index"] = new_global_indices
                else:
                    df_extracted["index"] = new_global_indices
            else:
                    df_extracted["index"] = new_global_indices

            # Write extracted parquet
            write_parquet_like_source(df_extracted, src_parquet_path, tgt_parquet_path)

        except Exception as e:
            warnings.append(f"Failed to extract parquet for episode {src_ep_idx}: {e}")
            continue

        # --- EXTRACT VIDEOS ---
        for vid_key in video_keys:
            try:
                src_vpath_rel = src_info["video_path"].format(
                    episode_chunk=src_chunk_idx,
                    video_key=vid_key,
                    episode_index=src_ep_idx
                )
                src_vpath = (src_root / src_vpath_rel).resolve()
                if not src_vpath.is_file():
                    alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                    if alt_vid:
                        src_vpath = alt_vid
                    else:
                        warnings.append(f"Video {vid_key} for episode {src_ep_idx} not found, skipping")
                        continue
            except Exception:
                alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                if alt_vid:
                    src_vpath = alt_vid
                else:
                    warnings.append(f"Video {vid_key} for episode {src_ep_idx} not found, skipping")
                    continue

            tgt_vchunk_idx = tgt_chunk_idx
            tgt_vpath_rel = meta_target.video_path.format(
                episode_chunk=tgt_vchunk_idx,
                video_key=vid_key,
                episode_index=new_ep_idx
            )
            tgt_vpath = meta_target.root / tgt_vpath_rel

            video_tasks.append({
                "label": f"{vid_key} episode {src_ep_idx}",
                "src_path": str(src_vpath),
                "tgt_path": str(tgt_vpath),
                "extraction_factor": extraction_factor,
                "source_length": src_ep_length,
                "target_length": new_ep_length,
            })

        # Get episode stats (may use string keys)
        ep_stats = {}
        if isinstance(src_episodes_stats, dict):
            ep_stats = src_episodes_stats.get(str(src_ep_idx), src_episodes_stats.get(src_ep_idx, {}))

        # Register episode in target meta
        meta_target.save_episode(
            episode_index=new_ep_idx,
            episode_length=new_ep_length,
            episode_tasks=ep_tasks,
            episode_stats=ep_stats
        )

    warnings.extend(process_video_scale_tasks(video_tasks, num_workers, video_config, "Extracting videos"))
    update_info_json_video_codec(meta_target.root, video_config)

    # Summary
    print("\n[+] Extraction finished!")
    print(f"    Target total_episodes: {meta_target.info.get('total_episodes')}")
    print(f"    Target total_frames  : {meta_target.info.get('total_frames')}")
    print(f"    Reduction factor     : ~{extraction_factor}x")

    if warnings:
        print(f"\n[!] {len(warnings)} warnings:")
        for w in warnings[:10]:  # Show first 10 warnings
            print("  -", w)
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")

    # Smoke test
    try:
        ds_check = LeRobotDataset(repo_id=repo_id, root=str(meta_target.root))
        print("\n[*] Smoke test succeeded!")
        print(f"    Loaded dataset: {ds_check.num_episodes} episodes, {ds_check.num_frames} frames")
    except Exception as e:
        print(f"\n[!] Warning: failed to load target dataset: {e}")

    # Merge with additional datasets if requested
    if merge_src_paths and len(merge_src_paths) > 0:
        if not MERGE_AVAILABLE:
            print("\n[!] Warning: merge_lerobot module not available. Skipping merge step.")
            return

        print("\n" + "=" * 60)
        print("Merging extracted dataset with additional sources")
        print("=" * 60)

        # Prepare merge source paths (include extracted dataset)
        all_merge_srcs = [tgt_path] + merge_src_paths
        merge_final_path = merge_tgt_path if merge_tgt_path else tgt_path + "_merged"
        merge_final_repo_id = merge_repo_id if merge_repo_id else repo_id + "_merged"

        print(f"Merge sources: {all_merge_srcs}")
        print(f"Merge target: {merge_final_path}")
        print(f"Merge repo_id: {merge_final_repo_id}")
        print()

        try:
            # Infer features/fps/robot_type from extracted dataset
            merge_repos(
                src_paths=all_merge_srcs,
                tgt_path=merge_final_path,
                repo_id=merge_final_repo_id,
                fps=fps,
                robot_type=robot_type,
                features=output_features,
                force=merge_force
            )
            print("\n[+] Merge complete!")
        except Exception as e:
            print(f"\n[!] Error during merge: {e}")
            raise

    return


def time_scaling_with_split(
    src_path: str,
    tgt_path: str,
    repo_id: str,
    split_ratio: float = 0.3,
    extraction_factor: int = 2,
    num_workers: int = 4,
    force: bool = False,
    video_config: Optional[VideoEncodeConfig] = None,
):
    """
    Split dataset by ratio, extract frames from one part, and merge both parts.

    Args:
        src_path: Path to source LeRobot dataset
        tgt_path: Path to target (final merged) dataset
        repo_id: Repository ID for the new dataset (will append _time_scaling suffix)
        split_ratio: Ratio of data to extract (e.g., 0.3 means 30% will be extracted, 70% kept original)
        extraction_factor: Extract every Nth frame for the extracted portion
        num_workers: Number of parallel worker processes for video processing
        force: Force operation even if target exists
    """
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")

    if extraction_factor < 1:
        raise ValueError(f"extraction_factor must be >= 1, got {extraction_factor}")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {num_workers}")
    video_config = video_config or VideoEncodeConfig()
    validate_video_encode_config(video_config, num_workers)

    src_root = Path(src_path).expanduser().resolve()
    if not src_root.exists():
        raise RuntimeError(f"Source path does not exist: {src_path}")

    tgt_root = Path(tgt_path).expanduser().resolve()
    if tgt_root.exists() and any(tgt_root.iterdir()) and not force:
        raise RuntimeError(f"Target {tgt_root} exists and is not empty. Use --force or remove it first.")

    # Load source dataset
    print(f"[*] Loading source dataset from {src_root}")
    try:
        src_info = load_info(src_root)
        src_episodes = load_episodes(src_root)
        src_tasks_dict, _ = load_tasks(src_root)
    except Exception as e:
        raise RuntimeError(f"Cannot load source dataset: {e}") from e

    try:
        src_episodes_stats = load_episodes_stats(src_root)
    except Exception:
        src_episodes_stats = {}

    # Extract parameters from source
    fps = src_info.get("fps", 30)
    robot_type = src_info.get("robot_type", "unknown")
    features = src_info.get("features", {})
    output_features = output_features_for_video_config(features, video_config)
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    total_episodes = len(src_episodes)
    extract_count = max(1, int(total_episodes * split_ratio))
    keep_count = total_episodes - extract_count

    print_time_scaling_plan(
        src_root=src_root,
        tgt_root=tgt_root,
        repo_id=repo_id,
        fps=float(fps),
        robot_type=robot_type,
        total_episodes=total_episodes,
        total_frames=int(src_info.get("total_frames", 0)),
        extraction_factor=extraction_factor,
        num_workers=num_workers,
        video_keys=video_keys,
        video_config=video_config,
        split_ratio=split_ratio,
        extract_count=extract_count,
        keep_count=keep_count,
    )

    print("[*] Source dataset info:")
    print(f"    - FPS: {fps}")
    print(f"    - Robot type: {robot_type}")
    print(f"    - Total episodes: {total_episodes}")
    print(f"    - Split ratio: {split_ratio}")
    print(f"    - Episodes to extract: {extract_count}")
    print(f"    - Episodes to keep original: {keep_count}")
    print(f"    - Extraction factor: {extraction_factor}")
    print(f"    - Video workers: {num_workers}")
    print(f"    - Video codec: {video_config.video_codec}")

    # Sort episodes by index
    items_sorted = sorted(src_episodes.items(), key=lambda kv: int(kv[0]))

    # Split episodes
    extract_episodes = dict(items_sorted[:extract_count])
    keep_episodes = dict(items_sorted[extract_count:])

    print("\n[*] Split episodes:")
    print(f"    - Extracting: episodes {list(extract_episodes.keys())[0]} to {list(extract_episodes.keys())[-1]}")
    print(f"    - Keeping original: episodes {list(keep_episodes.keys())[0]} to {list(keep_episodes.keys())[-1]}")

    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp(prefix="time_scaling_"))
    extract_tgt_path = str(temp_dir / "extracted")
    keep_tgt_path = str(temp_dir / "kept")

    try:
        # Step 1: Extract frames from first portion
        print(f"\n[1/3] Extracting frames from {extract_count} episodes...")
        # Create target dataset structure for extracted portion
        ds_extract = LeRobotDataset.create(
            repo_id=repo_id + "_extracted",
            fps=int(fps),
            root=extract_tgt_path,
            robot_type=robot_type,
            features=output_features
        )
        meta_extract = ds_extract.meta

        # Add tasks
        for _k, task in sorted(src_tasks_dict.items(), key=lambda kv: int(kv[0])):
            meta_extract.add_task(task)

        warnings: List[str] = []
        extract_video_tasks: List[Dict[str, Any]] = []

        # Process extract episodes
        for src_ep_idx_str, ep in tqdm(extract_episodes.items(), desc="Extracting episodes"):
            src_ep_idx = int(src_ep_idx_str)
            src_ep_length = int(ep.get("length", 0))
            ep_tasks = ep.get("tasks", [])

            # Calculate new episode length after extraction
            new_ep_length = (src_ep_length + extraction_factor - 1) // extraction_factor
            if new_ep_length == 0:
                warnings.append(f"Episode {src_ep_idx} too short ({src_ep_length} frames), skipping")
                continue

            # Find source parquet file
            src_chunksize = int(src_info.get("chunks_size", 1000))
            src_chunk_idx = src_ep_idx // src_chunksize

            try:
                src_parquet_rel = src_info["data_path"].format(
                    episode_chunk=src_chunk_idx,
                    episode_index=src_ep_idx
                )
                src_parquet_path = (src_root / src_parquet_rel).resolve()
                if not src_parquet_path.is_file():
                    alt = find_parquet_by_episode(src_root, src_ep_idx)
                    if alt:
                        src_parquet_path = alt
                    else:
                        raise FileNotFoundError
            except Exception:
                alt = find_parquet_by_episode(src_root, src_ep_idx)
                if alt:
                    src_parquet_path = alt
                else:
                    warnings.append(f"Parquet for episode {src_ep_idx} not found, skipping")
                    continue

            # New episode index in target
            new_ep_idx = int(meta_extract.info["total_episodes"])

            # Target paths
            tgt_chunk_idx = meta_extract.get_episode_chunk(new_ep_idx)
            tgt_parquet_rel = meta_extract.data_path.format(
                episode_chunk=tgt_chunk_idx,
                episode_index=new_ep_idx
            )
            tgt_parquet_path = meta_extract.root / tgt_parquet_rel

            # Extract & patch parquet
            start_global_index = int(meta_extract.info.get("total_frames", 0))
            tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                df = pd.read_parquet(str(src_parquet_path))
                total_frames = len(df)

                # Extract every Nth frame
                extracted_indices = list(range(0, total_frames, extraction_factor))
                df_extracted = df.iloc[extracted_indices].copy()
                df_extracted.reset_index(drop=True, inplace=True)

                n_rows = len(df_extracted)

                def make_episode_index_col(series, scalar_val, n):
                    if len(series) > 0 and series.dtype == object and isinstance(series.iloc[0], (list, tuple, np.ndarray)):
                        return pd.Series([[int(scalar_val)]] * n, index=range(n))
                    else:
                        return pd.Series([int(scalar_val)] * n, index=range(n))

                df_extracted["episode_index"] = make_episode_index_col(df_extracted["episode_index"], new_ep_idx, n_rows)

                # Update frame_index
                if "frame_index" in df_extracted.columns:
                    new_frame_indices = list(range(n_rows))
                    if df_extracted["frame_index"].dtype == object and n_rows > 0:
                        first_val = df_extracted["frame_index"].iloc[0]
                        if isinstance(first_val, (list, tuple, np.ndarray)):
                            df_extracted["frame_index"] = [[int(x)] for x in new_frame_indices]
                        else:
                            df_extracted["frame_index"] = new_frame_indices
                    else:
                        df_extracted["frame_index"] = new_frame_indices

                # Update timestamp
                if "timestamp" in df_extracted.columns:
                    frame_duration = 1.0 / fps
                    new_timestamps = (np.arange(n_rows, dtype=np.float32) * np.float32(frame_duration)).astype(np.float32)
                    if df_extracted["timestamp"].dtype == object and n_rows > 0:
                        first_val = df_extracted["timestamp"].iloc[0]
                        if isinstance(first_val, (list, tuple, np.ndarray)):
                            df_extracted["timestamp"] = [[float(x)] for x in new_timestamps]
                        else:
                            df_extracted["timestamp"] = new_timestamps
                    else:
                        df_extracted["timestamp"] = new_timestamps

                # Update global index
                new_global_indices = list(range(start_global_index, start_global_index + n_rows))
                if "index" in df_extracted.columns:
                    if df_extracted["index"].dtype == object and n_rows > 0:
                        first_val = df_extracted["index"].iloc[0]
                        if isinstance(first_val, (list, tuple, np.ndarray)):
                            df_extracted["index"] = [[int(x)] for x in new_global_indices]
                        else:
                            df_extracted["index"] = new_global_indices
                    else:
                        df_extracted["index"] = new_global_indices
                else:
                    df_extracted["index"] = new_global_indices

                write_parquet_like_source(df_extracted, src_parquet_path, tgt_parquet_path)
            except Exception as e:
                warnings.append(f"Failed to extract parquet for episode {src_ep_idx}: {e}")
                continue

            # Extract videos
            for vid_key in video_keys:
                try:
                    src_vpath_rel = src_info["video_path"].format(
                        episode_chunk=src_chunk_idx,
                        video_key=vid_key,
                        episode_index=src_ep_idx
                    )
                    src_vpath = (src_root / src_vpath_rel).resolve()
                    if not src_vpath.is_file():
                        alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                        if alt_vid:
                            src_vpath = alt_vid
                        else:
                            continue
                except Exception:
                    alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                    if alt_vid:
                        src_vpath = alt_vid
                    else:
                        continue

                tgt_vchunk_idx = tgt_chunk_idx
                tgt_vpath_rel = meta_extract.video_path.format(
                    episode_chunk=tgt_vchunk_idx,
                    video_key=vid_key,
                    episode_index=new_ep_idx
                )
                tgt_vpath = meta_extract.root / tgt_vpath_rel

                extract_video_tasks.append({
                    "label": f"{vid_key} episode {src_ep_idx}",
                    "src_path": str(src_vpath),
                    "tgt_path": str(tgt_vpath),
                    "extraction_factor": extraction_factor,
                    "source_length": src_ep_length,
                    "target_length": new_ep_length,
                })

            # Get episode stats
            ep_stats = {}
            if isinstance(src_episodes_stats, dict):
                ep_stats = src_episodes_stats.get(str(src_ep_idx), src_episodes_stats.get(src_ep_idx, {}))

            meta_extract.save_episode(
                episode_index=new_ep_idx,
                episode_length=new_ep_length,
                episode_tasks=ep_tasks,
                episode_stats=ep_stats
            )

        warnings.extend(process_video_scale_tasks(extract_video_tasks, num_workers, video_config, "Extracting videos"))
        update_info_json_video_codec(meta_extract.root, video_config)

        if warnings:
            print(f"  [!] {len(warnings)} warnings during extraction")

        # Step 2: Copy original episodes for second portion
        print(f"\n[2/3] Copying original {keep_count} episodes...")
        # Create a new dataset with only the kept episodes
        ds_keep = LeRobotDataset.create(
            repo_id=repo_id + "_kept",
            fps=int(fps),
            root=keep_tgt_path,
            robot_type=robot_type,
            features=output_features
        )
        meta_keep = ds_keep.meta

        # Add tasks
        for _k, task in sorted(src_tasks_dict.items(), key=lambda kv: int(kv[0])):
            meta_keep.add_task(task)

        copy_video_tasks: List[Dict[str, Any]] = []

        for src_ep_idx_str, ep in tqdm(keep_episodes.items(), desc="Copying episodes"):
            src_ep_idx = int(src_ep_idx_str)
            ep_length = int(ep.get("length", 0))
            ep_tasks = ep.get("tasks", [])

            # Find source parquet
            src_chunksize = int(src_info.get("chunks_size", 1000))
            src_chunk_idx = src_ep_idx // src_chunksize

            try:
                src_parquet_rel = src_info["data_path"].format(
                    episode_chunk=src_chunk_idx,
                    episode_index=src_ep_idx
                )
                src_parquet_path = (src_root / src_parquet_rel).resolve()
                if not src_parquet_path.is_file():
                    alt = find_parquet_by_episode(src_root, src_ep_idx)
                    if alt:
                        src_parquet_path = alt
                    else:
                        raise FileNotFoundError
            except Exception:
                alt = find_parquet_by_episode(src_root, src_ep_idx)
                if alt:
                    src_parquet_path = alt
                else:
                    continue

            # New episode index
            new_ep_idx = int(meta_keep.info["total_episodes"])
            tgt_chunk_idx = meta_keep.get_episode_chunk(new_ep_idx)
            tgt_parquet_rel = meta_keep.data_path.format(
                episode_chunk=tgt_chunk_idx,
                episode_index=new_ep_idx
            )
            tgt_parquet_path = meta_keep.root / tgt_parquet_rel

            start_global_index = int(meta_keep.info.get("total_frames", 0))
            tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy and patch parquet
            try:
                df = pd.read_parquet(str(src_parquet_path))
                n_rows = len(df)

                def make_episode_index_col(series, scalar_val, n):
                    if series.dtype == object and n > 0 and isinstance(series.iloc[0], (list, tuple, np.ndarray)):
                        return series.apply(lambda _: [int(scalar_val)])
                    else:
                        return pd.Series([int(scalar_val)] * n, index=series.index)

                if "episode_index" in df.columns:
                    df["episode_index"] = make_episode_index_col(df["episode_index"], new_ep_idx, n_rows)
                else:
                    df["episode_index"] = [int(new_ep_idx)] * n_rows

                new_global_indices = list(range(start_global_index, start_global_index + n_rows))
                if "index" in df.columns:
                    if df["index"].dtype == object and n_rows > 0 and isinstance(df["index"].iloc[0], (list, tuple, np.ndarray)):
                        df["index"] = [[int(x)] for x in new_global_indices]
                    else:
                        df["index"] = new_global_indices
                else:
                    df["index"] = new_global_indices

                write_parquet_like_source(df, src_parquet_path, tgt_parquet_path)
            except Exception:
                shutil.copy2(str(src_parquet_path), str(tgt_parquet_path))

            # Copy videos
            for vid_key in video_keys:
                try:
                    src_vpath_rel = src_info["video_path"].format(
                        episode_chunk=src_chunk_idx,
                        video_key=vid_key,
                        episode_index=src_ep_idx
                    )
                    src_vpath = (src_root / src_vpath_rel).resolve()
                    if not src_vpath.is_file():
                        alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                        if alt_vid:
                            src_vpath = alt_vid
                        else:
                            continue
                except Exception:
                    alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                    if alt_vid:
                        src_vpath = alt_vid
                    else:
                        continue

                tgt_vchunk_idx = tgt_chunk_idx
                tgt_vpath_rel = meta_keep.video_path.format(
                    episode_chunk=tgt_vchunk_idx,
                    video_key=vid_key,
                    episode_index=new_ep_idx
                )
                tgt_vpath = meta_keep.root / tgt_vpath_rel
                copy_video_tasks.append({
                    "label": f"{vid_key} episode {src_ep_idx}",
                    "src_path": str(src_vpath),
                    "tgt_path": str(tgt_vpath),
                    "extraction_factor": 1,
                    "source_length": ep_length,
                    "target_length": ep_length,
                })

            # Get episode stats
            ep_stats = {}
            if isinstance(src_episodes_stats, dict):
                ep_stats = src_episodes_stats.get(str(src_ep_idx), src_episodes_stats.get(src_ep_idx, {}))

            meta_keep.save_episode(
                episode_index=new_ep_idx,
                episode_length=ep_length,
                episode_tasks=ep_tasks,
                episode_stats=ep_stats
            )

        if video_config.video_codec == "source":
            warnings.extend(process_video_copy_tasks(copy_video_tasks, num_workers))
        else:
            warnings.extend(process_video_scale_tasks(copy_video_tasks, num_workers, video_config, "Re-encoding kept videos"))
        update_info_json_video_codec(meta_keep.root, video_config)

        # Step 3: Merge extracted and kept portions
        print("\n[3/3] Merging extracted and kept portions...")
        if not MERGE_AVAILABLE:
            raise RuntimeError("merge_lerobot module not available. Cannot perform merge operation.")

        final_repo_id = repo_id + "_time_scaling"
        merge_repos(
            src_paths=[extract_tgt_path, keep_tgt_path],
            tgt_path=tgt_path,
            repo_id=final_repo_id,
            fps=fps,
            robot_type=robot_type,
            features=output_features,
            force=force
        )
        update_info_json_video_codec(Path(tgt_path), video_config)

        print("\n[+] Time scaling complete!")
        print(f"    Final dataset: {tgt_path}")
        print(f"    Final repo_id: {final_repo_id}")

    finally:
        # Cleanup temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    return


def add_video_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = VideoEncodeConfig()
    parser.add_argument(
        "--video-codec",
        "--video_codec",
        dest="video_codec",
        default=defaults.video_codec,
        help="输出视频编码器；默认 h264_nvenc，传 source 表示沿用源编码",
    )
    parser.add_argument("--gop", type=int, default=defaults.gop, help="输出视频 GOP/关键帧间隔（默认 2）")
    parser.add_argument(
        "--b-frames",
        "--b_frames",
        dest="b_frames",
        type=int,
        default=defaults.b_frames,
        help="输出视频 B 帧数量（默认 0）",
    )
    parser.add_argument(
        "--nvenc-preset",
        "--nvenc_preset",
        dest="nvenc_preset",
        default=defaults.nvenc_preset,
        help="h264_nvenc/hevc_nvenc preset（默认 p4）",
    )
    parser.add_argument(
        "--nvenc-cq",
        "--nvenc_cq",
        dest="nvenc_cq",
        type=int,
        default=defaults.nvenc_cq,
        help="h264_nvenc/hevc_nvenc CQ（默认 23）",
    )
    parser.add_argument(
        "--av1-crf",
        "--av1_crf",
        dest="av1_crf",
        type=int,
        default=defaults.av1_crf,
        help="libaom-av1 CRF（默认 30）",
    )
    parser.add_argument(
        "--av1-cpu-used",
        "--av1_cpu_used",
        dest="av1_cpu_used",
        type=int,
        default=defaults.av1_cpu_used,
        help="libaom-av1 cpu-used（默认 8）",
    )
    parser.add_argument(
        "--mp4v-qscale",
        "--mp4v_qscale",
        dest="mp4v_qscale",
        type=int,
        default=defaults.mp4v_qscale,
        help="MP4V/MPEG4 qscale（默认 3）",
    )
    parser.add_argument(
        "--ffmpeg-loglevel",
        "--ffmpeg_loglevel",
        dest="ffmpeg_loglevel",
        default=defaults.ffmpeg_loglevel,
        help="FFmpeg 日志级别（默认 error）",
    )


def video_encode_config_from_args(args: argparse.Namespace) -> VideoEncodeConfig:
    return VideoEncodeConfig(
        video_codec=args.video_codec,
        gop=args.gop,
        b_frames=args.b_frames,
        nvenc_preset=args.nvenc_preset,
        nvenc_cq=args.nvenc_cq,
        av1_crf=args.av1_crf,
        av1_cpu_used=args.av1_cpu_used,
        mp4v_qscale=args.mp4v_qscale,
        ffmpeg_loglevel=args.ffmpeg_loglevel,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract (downsample) frames from a LeRobot dataset"
    )
    parser.add_argument(
        "--src-path",
        "--src_path",
        dest="src_path",
        required=True,
        help="Path to source LeRobot dataset"
    )
    parser.add_argument(
        "--tgt-path",
        "--tgt_path",
        dest="tgt_path",
        required=True,
        help="Path to target (extracted) dataset"
    )
    parser.add_argument(
        "--repo-id",
        "--repo_id",
        dest="repo_id",
        required=True,
        help="Repository ID for the new dataset"
    )
    parser.add_argument(
        "--extraction-factor",
        "--extraction_factor",
        dest="extraction_factor",
        type=int,
        default=2,
        help="Extract every Nth frame (default: 2, meaning keep frames 0, 2, 4, ...)"
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for video processing (default: 4)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force extraction even if target directory exists"
    )
    parser.add_argument(
        "--merge-src-paths",
        "--merge_src_paths",
        dest="merge_src_paths",
        nargs="+",
        default=None,
        help="Optional: Additional source dataset paths to merge with extracted dataset"
    )
    parser.add_argument(
        "--merge-tgt-path",
        "--merge_tgt_path",
        dest="merge_tgt_path",
        type=str,
        default=None,
        help="Optional: Path for merged dataset (default: <tgt_path>_merged)"
    )
    parser.add_argument(
        "--merge-repo-id",
        "--merge_repo_id",
        dest="merge_repo_id",
        type=str,
        default=None,
        help="Optional: Repository ID for merged dataset (default: <repo_id>_merged)"
    )
    parser.add_argument(
        "--merge-force",
        "--merge_force",
        dest="merge_force",
        action="store_true",
        help="Force merge even if conflicts exist"
    )
    parser.add_argument(
        "--split-ratio",
        "--split_ratio",
        dest="split_ratio",
        type=float,
        default=None,
        help="Enable split mode: ratio of data to extract (0.0-1.0). If set, splits dataset, extracts one portion, and merges both."
    )
    add_video_arguments(parser)

    args = parser.parse_args()

    # Check if split mode is enabled
    if args.split_ratio is not None:
        print("=" * 60)
        print("LeRobot Dataset Time Scaling (Split & Extract)")
        print("=" * 60)

        time_scaling_with_split(
            src_path=args.src_path,
            tgt_path=args.tgt_path,
            repo_id=args.repo_id,
            split_ratio=args.split_ratio,
            extraction_factor=args.extraction_factor,
            num_workers=args.num_workers,
            force=args.force,
            video_config=video_encode_config_from_args(args),
        )
    else:
        print("=" * 60)
        print("LeRobot Dataset Frame Extraction")
        print("=" * 60)

        extract_dataset(
            src_path=args.src_path,
            tgt_path=args.tgt_path,
            repo_id=args.repo_id,
            extraction_factor=args.extraction_factor,
            num_workers=args.num_workers,
            force=args.force,
            merge_src_paths=args.merge_src_paths,
            merge_tgt_path=args.merge_tgt_path,
            merge_repo_id=args.merge_repo_id,
            merge_force=args.merge_force,
            video_config=video_encode_config_from_args(args),
        )
