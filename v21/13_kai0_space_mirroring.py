#!/usr/bin/env python3
"""
space_mirror.py

Space Mirror core functionality: dual-arm data mirroring and data augmentation
- Swap left/right arm data (parquet, json, jsonl)
- Flip videos (horizontal mirroring)
- Merge original and mirrored datasets
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import merge function from merge_lerobot
# Add current directory to path to import merge_lerobot from same directory
_utils_dir = Path(__file__).parent
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))
try:
    from merge_lerobot import merge_repos
    MERGE_AVAILABLE = True
except ImportError:
    MERGE_AVAILABLE = False
    print("Warning: merge_lerobot module not available. Merge functionality will be disabled.")

# ==================== Core Utility Functions ====================

STATE_KEY_CANDIDATES = ("observation.state", "state")
ACTION_KEY_CANDIDATES = ("action", "actions")
ARM_DIM = 7
JOINT_COUNT = 6
DEFAULT_VIDEO_CAMERA_MAP = (
    ("observation.images.top_head", "observation.images.top_head"),
    ("observation.images.hand_right", "observation.images.hand_left"),
    ("observation.images.hand_left", "observation.images.hand_right"),
)
CAM_WRIST_VIDEO_CAMERA_MAP = (
    ("observation.images.cam_high", "observation.images.cam_high"),
    ("observation.images.cam_right_wrist", "observation.images.cam_left_wrist"),
    ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"),
)
SUPPORTED_VIDEO_CAMERA_MAPS = (
    DEFAULT_VIDEO_CAMERA_MAP,
    CAM_WRIST_VIDEO_CAMERA_MAP,
)
STATS_IMAGE_SWAP_PAIRS = (
    ("observation.images.hand_left", "observation.images.hand_right"),
    ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"),
)
META_FILES_TO_COPY = ("episodes.jsonl", "tasks.jsonl")
NVENC_MIN_SIDE_FOR_SCRIPT = 256


@dataclass(frozen=True)
class MirrorConfig:
    """Validated configuration shared by plan printing and all transformations."""

    state_key: str
    action_key: str
    negate_joints: Tuple[int, ...]
    state_key_source: str = "参数"
    action_key_source: str = "参数"
    norm_state_key: Optional[str] = None
    norm_action_key: Optional[str] = None
    left_dim: int = ARM_DIM
    right_dim: int = ARM_DIM

    @property
    def total_dim(self) -> int:
        return self.left_dim + self.right_dim

    @property
    def negate_offsets(self) -> Tuple[int, ...]:
        return tuple(joint - 1 for joint in self.negate_joints)

    @property
    def output_mapping(self) -> Tuple[Tuple[int, bool], ...]:
        """Return (zero-based input index, negate) for every output dimension."""
        mapping = []
        for output_index in range(self.total_dim):
            local_index = output_index % ARM_DIM
            input_index = (
                output_index + self.right_dim
                if output_index < self.left_dim
                else output_index - self.left_dim
            )
            mapping.append((input_index, local_index in self.negate_offsets))
        return tuple(mapping)


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
class VideoTask:
    source_path: Path
    target_path: Path
    source_key: str
    target_key: str
    episode_index: int
    expected_frames: int
    params: VideoParams
    encoding: ResolvedVideoEncoding


@dataclass(frozen=True)
class VideoTaskCandidate:
    source_path: Path
    target_path: Path
    source_key: str
    target_key: str
    episode_index: int
    expected_frames: int


def validate_mirror_options(
    left_dim: int,
    right_dim: int,
    negate_joints: Optional[Sequence[int]],
) -> Tuple[int, ...]:
    if left_dim != ARM_DIM or right_dim != ARM_DIM:
        raise ValueError(
            "当前仅支持左右臂各 7 维（6 个关节角 + 1 个夹爪）；"
            f"收到 left_dim={left_dim}, right_dim={right_dim}"
        )
    if not negate_joints:
        raise ValueError(
            "必须通过 --negate-joints 手动指定需要取反的关节轴；"
            "Piper 使用 1 4 6，PiperX 使用 1 5 6"
        )

    joints = tuple(int(joint) for joint in negate_joints)
    if len(set(joints)) != len(joints):
        raise ValueError(f"--negate-joints 不能包含重复轴: {list(joints)}")
    invalid = [joint for joint in joints if not 1 <= joint <= JOINT_COUNT]
    if invalid:
        raise ValueError(
            f"--negate-joints 只能包含 1..{JOINT_COUNT}，第 7 维夹爪不支持取反；"
            f"非法值: {invalid}"
        )
    return tuple(sorted(joints))


def _resolve_feature_key(
    features: Dict[str, Any],
    requested_key: Optional[str],
    candidates: Sequence[str],
    label: str,
) -> Tuple[str, str]:
    if requested_key is not None:
        if requested_key not in features:
            raise KeyError(
                f"meta/info.json 的 features 中不存在显式指定的 {label} key: {requested_key}"
            )
        return requested_key, "命令行参数"

    matches = [key for key in candidates if key in features]
    if len(matches) == 1:
        return matches[0], "meta/info.json 自动识别"
    if not matches:
        raise KeyError(
            f"无法从 meta/info.json 自动识别 {label} key；已尝试 {list(candidates)}，"
            f"请通过 --{label}-key 显式指定"
        )
    raise ValueError(
        f"meta/info.json 中存在多个候选 {label} key: {matches}；"
        f"请通过 --{label}-key 显式指定"
    )


def _validate_signal_feature(features: Dict[str, Any], key: str, label: str) -> None:
    feature = features.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"meta/info.json 中 {label} feature 不是对象: {key}")
    shape = feature.get("shape")
    if shape != [ARM_DIM * 2]:
        raise ValueError(
            f"当前仅支持 14 维双臂 {label}，meta/info.json 中 {key} 的 shape={shape}"
        )


def _resolve_norm_key(
    norm_stats: Dict[str, Any],
    feature_key: str,
    legacy_candidates: Sequence[str],
    label: str,
) -> Optional[str]:
    if feature_key in norm_stats:
        return feature_key
    matches = [key for key in legacy_candidates if key in norm_stats]
    if len(matches) <= 1:
        return matches[0] if matches else None
    raise ValueError(
        f"norm_stats.json 中存在多个候选 {label} key: {matches}，无法确定应转换哪一个"
    )


def resolve_mirror_config(
    src_root: Path,
    state_key: Optional[str],
    action_key: Optional[str],
    negate_joints: Optional[Sequence[int]],
    left_dim: int = ARM_DIM,
    right_dim: int = ARM_DIM,
) -> MirrorConfig:
    joints = validate_mirror_options(left_dim, right_dim, negate_joints)
    info_path = src_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"自动解析 state/action key 需要文件: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"meta/info.json 缺少 features 对象: {info_path}")
    resolved_state_key, state_source = _resolve_feature_key(
        features, state_key, STATE_KEY_CANDIDATES, "state"
    )
    resolved_action_key, action_source = _resolve_feature_key(
        features, action_key, ACTION_KEY_CANDIDATES, "action"
    )
    if resolved_state_key == resolved_action_key:
        raise ValueError("state key 和 action key 不能相同")
    _validate_signal_feature(features, resolved_state_key, "state")
    _validate_signal_feature(features, resolved_action_key, "action")

    norm_state_key = None
    norm_action_key = None
    norm_stats_path = src_root / "norm_stats.json"
    if norm_stats_path.is_file():
        with norm_stats_path.open("r", encoding="utf-8") as f:
            norm_data = json.load(f)
        norm_stats = norm_data.get("norm_stats")
        if not isinstance(norm_stats, dict):
            raise ValueError(f"norm_stats.json 缺少 norm_stats 对象: {norm_stats_path}")
        norm_state_key = _resolve_norm_key(
            norm_stats, resolved_state_key, ("state", "observation.state"), "state"
        )
        norm_action_key = _resolve_norm_key(
            norm_stats, resolved_action_key, ("actions", "action"), "action"
        )

    return MirrorConfig(
        state_key=resolved_state_key,
        action_key=resolved_action_key,
        negate_joints=joints,
        state_key_source=state_source,
        action_key_source=action_source,
        norm_state_key=norm_state_key,
        norm_action_key=norm_action_key,
        left_dim=left_dim,
        right_dim=right_dim,
    )


def swap_arms_in_array(arr: np.ndarray, config: MirrorConfig, negate: bool = True) -> np.ndarray:
    """Swap arm blocks and optionally negate configured joint axes."""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if arr.ndim == 0:
        raise ValueError("标量不能作为双臂数据进行镜像")

    arr_flat = arr.flatten()
    if len(arr_flat) != config.total_dim:
        raise ValueError(
            f"数组维度不匹配：期望 {config.total_dim} 维，实际 {len(arr_flat)} 维"
        )

    swapped = np.empty_like(arr_flat)
    for output_index, (input_index, should_negate) in enumerate(config.output_mapping):
        value = arr_flat[input_index]
        swapped[output_index] = -value if negate and should_negate else value

    if arr.ndim > 1:
        swapped = swapped.reshape(arr.shape)

    return swapped


def swap_array_dims_list(
    arr: Sequence[float],
    config: MirrorConfig,
    negate: bool = True,
) -> List[float]:
    """List variant of the configured arm transformation."""
    return swap_arms_in_array(np.asarray(arr), config, negate=negate).tolist()


def _transform_bound_pair(
    lower: Sequence[float],
    upper: Sequence[float],
    config: MirrorConfig,
) -> Tuple[List[float], List[float]]:
    swapped_lower = np.asarray(swap_array_dims_list(lower, config, negate=False))
    swapped_upper = np.asarray(swap_array_dims_list(upper, config, negate=False))
    result_lower = swapped_lower.copy()
    result_upper = swapped_upper.copy()
    for output_index, (_, should_negate) in enumerate(config.output_mapping):
        if should_negate:
            result_lower[output_index] = -swapped_upper[output_index]
            result_upper[output_index] = -swapped_lower[output_index]
    return result_lower.tolist(), result_upper.tolist()


def transform_stats_item(
    stat_item: Dict[str, Any],
    config: MirrorConfig,
    bound_pairs: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    """Mirror vector statistics while preserving their mathematical meaning."""
    if "mean" in stat_item:
        stat_item["mean"] = swap_array_dims_list(stat_item["mean"], config, negate=True)
    if "std" in stat_item:
        stat_item["std"] = swap_array_dims_list(stat_item["std"], config, negate=False)

    for lower_key, upper_key in bound_pairs:
        present = (lower_key in stat_item, upper_key in stat_item)
        if present == (False, False):
            continue
        if present != (True, True):
            raise ValueError(f"统计量 {lower_key}/{upper_key} 必须成对存在")
        lower, upper = _transform_bound_pair(
            stat_item[lower_key], stat_item[upper_key], config
        )
        stat_item[lower_key] = lower
        stat_item[upper_key] = upper
    return stat_item


# ==================== Parquet Processing ====================

def swap_arms_in_parquet(
    input_path: Path,
    output_path: Path,
    config: MirrorConfig,
) -> Tuple[str, bool, str]:
    """Mirror state and action in one parquet file."""
    try:
        df = pd.read_parquet(str(input_path))

        columns_to_process = [config.state_key, config.action_key]
        missing_columns = [col for col in columns_to_process if col not in df.columns]
        if missing_columns:
            return (
                str(input_path),
                False,
                f"缺少待转换列 {missing_columns}；实际列: {list(df.columns)}",
            )

        for col in columns_to_process:
            if df[col].dtype != object:
                return (
                    str(input_path),
                    False,
                    f"列 {col} 不是 object 类型，无法按嵌套数组处理",
                )

            swapped_values = []
            for idx, val in enumerate(df[col]):
                try:
                    if isinstance(val, (list, tuple)):
                        arr = np.array(val)
                    elif isinstance(val, np.ndarray):
                        arr = val.copy()
                    else:
                        return (
                            str(input_path),
                            False,
                            f"列 {col} 第 {idx} 行的数据类型不受支持: {type(val)}",
                        )

                    swapped_arr = swap_arms_in_array(arr, config, negate=True)
                    swapped_values.append(swapped_arr)

                except Exception as e:
                    return (
                        str(input_path),
                        False,
                        f"转换列 {col} 第 {idx} 行失败: {e}",
                    )

            df[col] = swapped_values

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(output_path), index=False)

        return (
            str(input_path),
            True,
            f"已转换 {len(columns_to_process)} 列: {', '.join(columns_to_process)}",
        )

    except Exception as e:
        return (str(input_path), False, f"错误: {e}")


def process_parquet_files(
    input_dir: Path,
    output_dir: Path,
    config: MirrorConfig,
    num_workers: int = 4,
) -> None:
    """Batch process parquet files"""
    parquet_files = list(input_dir.rglob('*.parquet'))

    if not parquet_files:
        print(f"Warning: No parquet files found in {input_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files")

    def get_output_path(input_file: Path) -> Path:
        relative = input_file.relative_to(input_dir)
        return output_dir / relative

    tasks = [(f, get_output_path(f)) for f in parquet_files]

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(swap_arms_in_parquet, inp, out, config): inp
            for inp, out in tasks
        }

        for future in tqdm(as_completed(future_to_file), total=len(tasks), desc="Processing parquet"):
            input_path = future_to_file[future]
            try:
                result_path, success, message = future.result()
                if success:
                    success_count += 1
                else:
                    print(f"✗ [{result_path}] {message}")
                    fail_count += 1
            except Exception as e:
                print(f"✗ [{input_path}] Processing exception: {str(e)}")
                fail_count += 1

    print(f"Parquet processing complete: {success_count} succeeded, {fail_count} failed")
    if fail_count:
        raise RuntimeError(f"{fail_count} 个 parquet 文件转换失败")


# ==================== JSON Processing ====================

def process_norm_stats_json(
    input_path: Path,
    output_path: Path,
    config: MirrorConfig,
) -> Tuple[str, bool, str]:
    """Process norm_stats.json file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "norm_stats" not in data:
            return (str(input_path), False, "Field 'norm_stats' not found in JSON file")

        norm_stats = data["norm_stats"]

        processed_keys = []
        for key in (config.norm_state_key, config.norm_action_key):
            if key is None:
                continue
            stat_item = norm_stats.get(key)
            if not isinstance(stat_item, dict):
                return (str(input_path), False, f"norm_stats.{key} 不是统计对象")
            transform_stats_item(stat_item, config, (("q01", "q99"),))
            processed_keys.append(key)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return (str(input_path), True, f"已转换统计键: {processed_keys or '无'}")

    except Exception as e:
        return (str(input_path), False, f"错误: {e}")


# ==================== JSONL Processing ====================

def swap_stats_dims(
    stats_dict: Dict[str, Any],
    config: MirrorConfig,
) -> Dict[str, Any]:
    """Swap left/right arm data in stats dictionary"""
    missing = object()
    for left_image_key, right_image_key in STATS_IMAGE_SWAP_PAIRS:
        left_stats = stats_dict.pop(left_image_key, missing)
        right_stats = stats_dict.pop(right_image_key, missing)
        if right_stats is not missing:
            stats_dict[left_image_key] = right_stats
        if left_stats is not missing:
            stats_dict[right_image_key] = left_stats

    for key in (config.state_key, config.action_key):
        if key not in stats_dict:
            raise KeyError(f"episode stats 缺少键: {key}")
        stat_item = stats_dict[key]
        if not isinstance(stat_item, dict):
            raise ValueError(f"episode stats 中 {key} 不是统计对象")
        transform_stats_item(stat_item, config, (("min", "max"),))

    return stats_dict


def process_episodes_stats_jsonl(
    input_path: Path,
    output_path: Path,
    config: MirrorConfig,
) -> Tuple[str, bool, str]:
    """Process episodes_stats.jsonl file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        processed_lines = []
        processed_count = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                processed_lines.append('')
                continue

            try:
                data = json.loads(line)

                if "stats" in data and isinstance(data["stats"], dict):
                    data["stats"] = swap_stats_dims(data["stats"], config)
                    processed_count += 1

                processed_lines.append(json.dumps(data, ensure_ascii=False))

            except json.JSONDecodeError as e:
                return (str(input_path), False, f"第 {line_num} 行 JSON 解析失败: {e}")
            except Exception as e:
                return (str(input_path), False, f"第 {line_num} 行转换失败: {e}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
            if processed_lines and processed_lines[-1]:
                f.write('\n')

        return (str(input_path), True, f"Successfully processed {processed_count} entries")

    except Exception as e:
        return (str(input_path), False, f"Error: {str(e)}")


# ==================== Video Processing ====================

def validate_video_encode_config(config: VideoEncodeConfig, num_workers: int) -> None:
    if not config.video_codec.strip():
        raise ValueError("--video-codec 不能为空")
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
    if probe_gop:
        keyframe_timestamps = probe_keyframe_timestamps(video_path)
    else:
        keyframe_timestamps = []
    if len(keyframe_timestamps) >= 2:
        gaps = [
            end - start
            for start, end in zip(
                keyframe_timestamps, keyframe_timestamps[1:], strict=False
            )
            if end > start
        ]
        if gaps:
            gop = max(1, int(round(float(np.median(gaps)) * fps)))
    return VideoParams(
        codec_name=canonicalize_codec(
            stream.get("codec_name", ""), stream.get("codec_tag_string", "")
        ),
        codec_tag=stream.get("codec_tag_string", ""),
        pix_fmt=stream.get("pix_fmt", "yuv420p"),
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=fps,
        fps_expr=fps_expr,
        gop=gop,
        b_frames=int(stream.get("has_b_frames", 0) or 0),
    )


def output_codec_name(
    params: VideoParams,
    config: VideoEncodeConfig,
) -> Tuple[str, bool]:
    requested_codec = (
        params.codec_name
        if config.video_codec == "source"
        else config.video_codec.strip()
    )
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


def resolve_video_encoding(
    params: VideoParams,
    config: VideoEncodeConfig,
) -> ResolvedVideoEncoding:
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


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def format_dataset_path(
    template: str,
    episode_index: int,
    chunks_size: int,
    video_key: Optional[str] = None,
) -> Path:
    values: Dict[str, Any] = {
        "episode_chunk": episode_chunk(episode_index, chunks_size),
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return Path(template.format(**values))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def video_feature_keys(info: Dict[str, Any]) -> Tuple[str, ...]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("meta/info.json 的 features 必须是对象")
    return tuple(
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


def resolve_video_camera_map(info: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    video_keys = set(video_feature_keys(info))
    if not video_keys:
        return ()

    matches = []
    for camera_map in SUPPORTED_VIDEO_CAMERA_MAPS:
        source_keys = {source_key for source_key, _target_key in camera_map}
        selected = tuple(
            (source_key, target_key)
            for source_key, target_key in camera_map
            if source_key in video_keys
        )
        if not selected:
            continue
        unknown_keys = sorted(video_keys - source_keys)
        matches.append((len(selected), camera_map, selected, unknown_keys))

    if not matches:
        supported_keys = sorted(
            {source_key for camera_map in SUPPORTED_VIDEO_CAMERA_MAPS for source_key, _ in camera_map}
        )
        raise ValueError(
            "meta/info.json 声明了 video feature，但没有可用的相机镜像映射。\n"
            f"  数据集 video keys: {sorted(video_keys)}\n"
            f"  当前支持: {supported_keys}"
        )

    best_count = max(match[0] for match in matches)
    best_matches = [match for match in matches if match[0] == best_count]
    if len(best_matches) > 1:
        raise ValueError(
            "video 相机映射不明确，请统一相机命名后重试。\n"
            f"  数据集 video keys: {sorted(video_keys)}"
        )

    _count, _camera_map, selected, unknown_keys = best_matches[0]
    if unknown_keys:
        raise ValueError(
            "meta/info.json 包含当前不支持的 video feature；为避免生成缺视频的数据集，已停止。\n"
            f"  不支持的 video keys: {unknown_keys}\n"
            f"  已识别映射: {list(selected)}"
        )
    return selected


def build_video_tasks(
    src_root: Path,
    tgt_root: Path,
    info: Dict[str, Any],
    episodes: Sequence[Dict[str, Any]],
    config: VideoEncodeConfig,
    video_camera_map: Optional[Sequence[Tuple[str, str]]] = None,
    num_workers: int = 1,
    show_progress: bool = False,
) -> List[VideoTask]:
    camera_mappings = tuple(video_camera_map) if video_camera_map is not None else resolve_video_camera_map(info)
    if not camera_mappings:
        return []

    video_path_template = info.get("video_path")
    if not isinstance(video_path_template, str) or not video_path_template:
        raise ValueError("meta/info.json 声明了视频 feature，但缺少 video_path 模板")
    chunks_size = int(info.get("chunks_size", 0))
    if chunks_size <= 0:
        raise ValueError(f"meta/info.json 的 chunks_size 无效: {chunks_size}")

    candidates: List[VideoTaskCandidate] = []
    for episode in episodes:
        if "episode_index" not in episode or "length" not in episode:
            raise ValueError(f"episodes.jsonl 缺少 episode_index/length: {episode}")
        episode_index = int(episode["episode_index"])
        expected_frames = int(episode["length"])
        if expected_frames < 0:
            raise ValueError(
                f"episode {episode_index} 的 length 不能为负数: {expected_frames}"
            )
        if expected_frames == 0:
            continue
        for source_key, target_key in camera_mappings:
            source_path = src_root / format_dataset_path(
                video_path_template, episode_index, chunks_size, source_key
            )
            target_path = tgt_root / format_dataset_path(
                video_path_template, episode_index, chunks_size, target_key
            )
            candidates.append(
                VideoTaskCandidate(
                    source_path=source_path,
                    target_path=target_path,
                    source_key=source_key,
                    target_key=target_key,
                    episode_index=episode_index,
                    expected_frames=expected_frames,
                )
            )

    missing = [candidate.source_path for candidate in candidates if not candidate.source_path.is_file()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n  ... 另有 {len(missing) - 20} 个"
        raise FileNotFoundError(f"发现缺失视频:\n{preview}{suffix}")

    ensure_video_tools()
    max_workers = max(1, min(int(num_workers), len(candidates) or 1))
    if show_progress and candidates:
        print(f"正在预检视频文件: files={len(candidates)}, workers={max_workers}")

    tasks: List[Optional[VideoTask]] = [None] * len(candidates)
    if max_workers == 1:
        iterable = candidates
        if show_progress:
            iterable = tqdm(candidates, desc="Probing videos")
        for index, candidate in enumerate(iterable):
            tasks[index] = build_video_task_from_candidate(candidate, config)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(build_video_task_from_candidate, candidate, config): index
                for index, candidate in enumerate(candidates)
            }
            try:
                futures = as_completed(future_to_index)
                if show_progress:
                    futures = tqdm(
                        futures,
                        total=len(future_to_index),
                        desc="Probing videos",
                    )
                for future in futures:
                    tasks[future_to_index[future]] = future.result()
            except Exception:
                for pending in future_to_index:
                    pending.cancel()
                raise

    resolved_tasks = [task for task in tasks if task is not None]

    required_encoders = {task.encoding.encoder for task in resolved_tasks}
    missing_encoders = sorted(
        encoder for encoder in required_encoders if not ffmpeg_has_encoder(encoder)
    )
    if missing_encoders:
        raise RuntimeError(f"当前 ffmpeg 不支持编码器: {', '.join(missing_encoders)}")
    return resolved_tasks


def build_video_task_from_candidate(
    candidate: VideoTaskCandidate,
    config: VideoEncodeConfig,
) -> VideoTask:
    source_frames = probe_video_frames(candidate.source_path)
    if source_frames != candidate.expected_frames:
        raise ValueError(
            f"源视频帧数与 episode 长度不一致: {candidate.source_path}, "
            f"episode={candidate.expected_frames}, video={source_frames}"
        )
    params = probe_video_params(candidate.source_path)
    return VideoTask(
        source_path=candidate.source_path,
        target_path=candidate.target_path,
        source_key=candidate.source_key,
        target_key=candidate.target_key,
        episode_index=candidate.episode_index,
        expected_frames=candidate.expected_frames,
        params=params,
        encoding=resolve_video_encoding(params, config),
    )


def build_ffmpeg_mirror_command(
    task: VideoTask,
    tmp_path: Path,
    config: VideoEncodeConfig,
) -> List[str]:
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        config.ffmpeg_loglevel,
        "-i",
        str(task.source_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "hflip,setpts=N/FRAME_RATE/TB",
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


def mirror_video_task(task: VideoTask, config: VideoEncodeConfig) -> None:
    task.target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"{task.target_path.stem}.mirror_",
        suffix=".mp4",
        dir=task.target_path.parent,
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink()
    cmd = build_ffmpeg_mirror_command(task, tmp_path, config)
    try:
        subprocess.run(cmd, check=True)
        actual_frames = probe_video_frames(tmp_path, exact=True)
        if actual_frames != task.expected_frames:
            raise RuntimeError(
                f"视频镜像后帧数不匹配: {task.source_path}, "
                f"期望 {task.expected_frames}, 实际 {actual_frames}"
            )
        tmp_path.replace(task.target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def process_video_tasks(
    tasks: Sequence[VideoTask],
    config: VideoEncodeConfig,
    num_workers: int,
) -> None:
    if not tasks:
        print("  Skipping: 没有需要处理的视频任务")
        return
    print(
        f"开始并行生成镜像视频: files={len(tasks)}, workers={num_workers}, "
        f"video_codec={config.video_codec}"
    )
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(mirror_video_task, task, config): task for task in tasks
        }
        try:
            for future in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="Processing videos",
            ):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as exc:
                    for pending in future_to_task:
                        pending.cancel()
                    raise RuntimeError(
                        f"视频生成失败，停止处理: episode={task.episode_index}, "
                        f"video={task.source_path}"
                    ) from exc
        finally:
            for future in future_to_task:
                if not future.done():
                    future.cancel()
    print(f"Video processing complete: {len(tasks)} succeeded, 0 failed")


def build_output_info(
    info: Dict[str, Any],
    config: VideoEncodeConfig,
    video_camera_map: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    output_info = dict(info)
    source_features = info.get("features", {})
    output_features = dict(source_features)
    camera_mappings = tuple(video_camera_map) if video_camera_map is not None else resolve_video_camera_map(info)
    mapped_keys = {key for mapping in camera_mappings for key in mapping}
    for key in mapped_keys:
        output_features.pop(key, None)
    for source_key, target_key in camera_mappings:
        feature = source_features.get(source_key)
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            output_features[target_key] = json.loads(json.dumps(feature))

    if config.video_codec != "source":
        codec_name = {
            "h264_nvenc": "h264",
            "hevc_nvenc": "hevc",
        }.get(config.video_codec, config.video_codec)
        for key, feature in list(output_features.items()):
            if not isinstance(feature, dict) or feature.get("dtype") != "video":
                continue
            updated_feature = dict(feature)
            video_info = dict(updated_feature.get("info", {}))
            video_info["video.codec"] = codec_name
            updated_feature["info"] = video_info
            output_features[key] = updated_feature

    output_info["features"] = output_features
    return output_info


# ==================== Dataset Merging ====================

def merge_lerobot_datasets(
    src_paths: List[str],
    tgt_path: str,
    repo_id: str,
    fps: int = 30,
    robot_type: str = "agilex",
    features: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> None:
    """Merge multiple LeRobot datasets by calling merge_lerobot.merge_repos"""
    if not MERGE_AVAILABLE:
        raise RuntimeError("merge_lerobot module not available. Cannot perform merge operation.")

    merge_repos(
        src_paths=src_paths,
        tgt_path=tgt_path,
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        force=force
    )


# ==================== Main Functions ====================

def _dimension_label(index: int) -> str:
    arm = "左臂" if index < ARM_DIM else "右臂"
    local_index = index % ARM_DIM
    if local_index < JOINT_COUNT:
        return f"{arm}关节{local_index + 1}"
    return f"{arm}夹爪"


def print_conversion_plan(
    src_root: Path,
    tgt_root: Path,
    config: MirrorConfig,
    video_config: Optional[VideoEncodeConfig] = None,
    video_tasks: Sequence[VideoTask] = (),
    video_camera_map: Optional[Sequence[Tuple[str, str]]] = None,
    merge_path: Optional[Path] = None,
) -> None:
    """Print the exact transformation represented by MirrorConfig."""
    video_config = video_config or VideoEncodeConfig()
    print("=" * 72)
    print("空间镜像转换方案（以下映射与实际转换共用同一配置）")
    print("=" * 72)
    print(f"源数据集: {src_root}")
    print(f"镜像数据集: {tgt_root}")
    print(f"State key: {config.state_key}（{config.state_key_source}）")
    print(f"Action key: {config.action_key}（{config.action_key_source}）")
    print(f"数据布局: 左臂 {config.left_dim} 维 + 右臂 {config.right_dim} 维")
    print(f"取反关节（从 1 开始）: {list(config.negate_joints)}")
    print("夹爪处理: 左右交换，不取反")
    print()
    print("State/Action 逐维映射:")
    for output_index, (input_index, negate) in enumerate(config.output_mapping):
        sign = "-" if negate else "+"
        print(
            f"  输出[{output_index + 1:2d}] {_dimension_label(output_index):<7} "
            f"= {sign}输入[{input_index + 1:2d}] {_dimension_label(input_index)}"
        )
    print()
    print("统计量映射:")
    print("  mean: 使用上述逐维映射")
    print("  std: 仅左右交换，不取反")
    print("  min/max: 取反轴使用 new_min=-old_max, new_max=-old_min")
    print("  q01/q99: 取反轴使用 new_q01=-old_q99, new_q99=-old_q01")
    print(
        "  norm_stats key: "
        f"state={config.norm_state_key or '未找到，保持原样'}, "
        f"action={config.norm_action_key or '未找到，保持原样'}"
    )
    print()
    print("视频映射（每帧水平翻转）:")
    camera_mappings = tuple(video_camera_map) if video_camera_map is not None else DEFAULT_VIDEO_CAMERA_MAP
    if camera_mappings:
        for source_key, target_key in camera_mappings:
            print(f"  {source_key} -> {target_key}")
    else:
        print("  无 video feature")
    print("  FFmpeg filter: hflip,setpts=N/FRAME_RATE/TB")
    print(f"  视频任务数: {len(video_tasks)}")
    print(f"  请求 codec: {video_config.video_codec}")
    print(f"  GOP/B frames: {video_config.gop}/{output_b_frames(video_config)}")
    if video_tasks:
        encoding_summaries = sorted(
            {
                (
                    task.params.codec_name,
                    task.encoding.codec,
                    task.encoding.encoder,
                    task.encoding.pix_fmt,
                    task.params.width,
                    task.params.height,
                )
                for task in video_tasks
            }
        )
        for source_codec, codec, encoder, pix_fmt, width, height in encoding_summaries:
            print(
                f"  编码方案: {source_codec} -> {codec} ({encoder}), "
                f"pix_fmt={pix_fmt}, size={width}x{height}"
            )
        fallback_count = sum(
            task.encoding.used_nvenc_fallback for task in video_tasks
        )
        if fallback_count:
            print(
                f"  NVENC 小分辨率回退: {fallback_count} 个任务使用 CPU 编码器"
            )
    print("  校验: 输出帧数必须等于 episodes.jsonl 中对应 episode.length")
    print("  写入: 目标目录临时 MP4 校验成功后原子替换")
    print("元数据复制: " + ", ".join(f"meta/{name}" for name in META_FILES_TO_COPY))
    print("元数据重建: meta/info.json（相机 feature 映射及 video.codec 同步）")
    if merge_path is not None:
        print(f"Full 模式合并: [{src_root}, {tgt_root}] -> {merge_path}")
    print("=" * 72)
    print()


def create_mirror_dataset(
    src_path: str,
    tgt_path: str,
    left_dim: int = 7,
    right_dim: int = 7,
    num_workers: int = 4,
    state_key: Optional[str] = None,
    action_key: Optional[str] = None,
    negate_joints: Optional[Sequence[int]] = None,
    video_config: Optional[VideoEncodeConfig] = None,
    merge_path: Optional[str] = None,
) -> None:
    """Create mirrored dataset"""
    src_root = Path(src_path).expanduser().resolve()
    tgt_root = Path(tgt_path).expanduser().resolve()

    if not src_root.is_dir():
        raise RuntimeError(f"Source dataset directory does not exist: {src_root}")

    config = resolve_mirror_config(
        src_root=src_root,
        state_key=state_key,
        action_key=action_key,
        negate_joints=negate_joints,
        left_dim=left_dim,
        right_dim=right_dim,
    )
    video_config = video_config or VideoEncodeConfig()
    validate_video_encode_config(video_config, num_workers)
    info_path = src_root / "meta" / "info.json"
    episodes_path = src_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"缺少数据集元数据: {episodes_path}")
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    episodes = read_jsonl(episodes_path)
    video_camera_map = resolve_video_camera_map(info)
    video_tasks = build_video_tasks(
        src_root=src_root,
        tgt_root=tgt_root,
        info=info,
        episodes=episodes,
        config=video_config,
        video_camera_map=video_camera_map,
        num_workers=num_workers,
        show_progress=True,
    )
    output_info = build_output_info(info, video_config, video_camera_map)
    resolved_merge_path = Path(merge_path).expanduser().resolve() if merge_path else None
    print_conversion_plan(
        src_root,
        tgt_root,
        config,
        video_config=video_config,
        video_tasks=video_tasks,
        video_camera_map=video_camera_map,
        merge_path=resolved_merge_path,
    )

    print("=" * 60)
    print("Starting to create mirrored dataset")
    print("=" * 60)
    print(f"Source path: {src_root}")
    print(f"Target path: {tgt_root}")
    print()

    # 1. Process norm_stats.json
    print("[1/5] Processing norm_stats.json...")
    src_norm_stats = src_root / "norm_stats.json"
    if src_norm_stats.exists():
        tgt_norm_stats = tgt_root / "norm_stats.json"
        _, success, msg = process_norm_stats_json(src_norm_stats, tgt_norm_stats, config)
        if success:
            print(f"✓ norm_stats.json processing complete: {msg}")
        else:
            raise RuntimeError(f"norm_stats.json processing failed: {msg}")
    else:
        print("  Skipping: norm_stats.json does not exist")
    print()

    # 2. Process episodes_stats.jsonl
    print("[2/5] Processing episodes_stats.jsonl...")
    src_episodes_stats = src_root / "meta" / "episodes_stats.jsonl"
    if src_episodes_stats.exists():
        tgt_episodes_stats = tgt_root / "meta" / "episodes_stats.jsonl"
        _, success, msg = process_episodes_stats_jsonl(
            src_episodes_stats, tgt_episodes_stats, config
        )
        if success:
            print(f"✓ episodes_stats.jsonl processing complete: {msg}")
        else:
            raise RuntimeError(f"episodes_stats.jsonl processing failed: {msg}")
    else:
        print("  Skipping: episodes_stats.jsonl does not exist")
    print()

    # 3. Process parquet files
    print("[3/5] Processing parquet files...")
    src_data_dir = src_root / "data"
    if src_data_dir.exists():
        tgt_data_dir = tgt_root / "data"
        process_parquet_files(src_data_dir, tgt_data_dir, config, num_workers)
        print("✓ All parquet files processing complete")
    else:
        print("  Skipping: data directory does not exist")
    print()

    # 4. Process video files
    print("[4/5] Flipping video files...")
    process_video_tasks(video_tasks, video_config, num_workers)
    print("✓ All video files processing complete")
    print()

    # 5. Copy other meta files
    print("[5/5] Copying other meta files...")
    src_meta_dir = src_root / "meta"
    if src_meta_dir.exists():
        tgt_meta_dir = tgt_root / "meta"
        tgt_meta_dir.mkdir(parents=True, exist_ok=True)

        for meta_file in META_FILES_TO_COPY:
            src_file = src_meta_dir / meta_file
            if src_file.exists():
                shutil.copy2(src_file, tgt_meta_dir / meta_file)
                print(f"  ✓ Copied {meta_file}")
        with (tgt_meta_dir / "info.json").open("w", encoding="utf-8") as f:
            json.dump(output_info, f, ensure_ascii=False, indent=4)
            f.write("\n")
        print("  ✓ Rebuilt info.json")

    print("✓ Mirrored dataset creation complete")
    print("=" * 60)


def _add_mirror_arguments(parser: argparse.ArgumentParser) -> None:
    video_defaults = VideoEncodeConfig()
    parser.add_argument(
        "--state-key",
        default=None,
        help=(
            "State 列/feature key；默认从 meta/info.json 自动识别 "
            "observation.state 或 state"
        ),
    )
    parser.add_argument(
        "--action-key",
        default=None,
        help=(
            "Action 列/feature key；默认从 meta/info.json 自动识别 action 或 actions"
        ),
    )
    parser.add_argument(
        "--negate-joints",
        type=int,
        nargs="+",
        required=True,
        metavar="AXIS",
        help=(
            "必须手动指定需要取反的关节轴（从 1 开始，仅限 1..6）；"
            "Piper: 1 4 6，PiperX: 1 5 6，无默认值"
        ),
    )
    parser.add_argument(
        "--left-dim",
        type=int,
        default=ARM_DIM,
        help="左臂维度；当前只允许 7（6 个关节角 + 1 个夹爪）",
    )
    parser.add_argument(
        "--right-dim",
        type=int,
        default=ARM_DIM,
        help="右臂维度；当前只允许 7（6 个关节角 + 1 个夹爪）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Parquet 进程池和 FFmpeg 视频线程池的并行数（默认 4）",
    )
    parser.add_argument(
        "--video-codec",
        "--video_codec",
        dest="video_codec",
        default=video_defaults.video_codec,
        help="输出视频编码器；默认 h264_nvenc，传 source 表示沿用源编码",
    )
    parser.add_argument(
        "--gop",
        type=int,
        default=video_defaults.gop,
        help="输出视频 GOP/关键帧间隔（默认 2）",
    )
    parser.add_argument(
        "--b-frames",
        "--b_frames",
        dest="b_frames",
        type=int,
        default=video_defaults.b_frames,
        help="输出视频 B 帧数量（默认 0）",
    )
    parser.add_argument(
        "--nvenc-preset",
        "--nvenc_preset",
        dest="nvenc_preset",
        default=video_defaults.nvenc_preset,
        help="h264_nvenc/hevc_nvenc preset（默认 p4）",
    )
    parser.add_argument(
        "--nvenc-cq",
        "--nvenc_cq",
        dest="nvenc_cq",
        type=int,
        default=video_defaults.nvenc_cq,
        help="h264_nvenc/hevc_nvenc CQ（默认 23）",
    )
    parser.add_argument(
        "--av1-crf",
        "--av1_crf",
        dest="av1_crf",
        type=int,
        default=video_defaults.av1_crf,
        help="libaom-av1 CRF（默认 30）",
    )
    parser.add_argument(
        "--av1-cpu-used",
        "--av1_cpu_used",
        dest="av1_cpu_used",
        type=int,
        default=video_defaults.av1_cpu_used,
        help="libaom-av1 cpu-used（默认 8）",
    )
    parser.add_argument(
        "--mp4v-qscale",
        "--mp4v_qscale",
        dest="mp4v_qscale",
        type=int,
        default=video_defaults.mp4v_qscale,
        help="MP4V/MPEG4 qscale（默认 3）",
    )
    parser.add_argument(
        "--ffmpeg-loglevel",
        "--ffmpeg_loglevel",
        dest="ffmpeg_loglevel",
        default=video_defaults.ffmpeg_loglevel,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Space Mirror: Dual-arm data mirroring and data augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Piper: create mirrored dataset
  python space_mirror.py create-mirror --src-path /path/to/original --tgt-path /path/to/mirror --negate-joints 1 4 6

  # PiperX: create mirrored dataset
  python space_mirror.py create-mirror --src-path /path/to/original --tgt-path /path/to/mirror --negate-joints 1 5 6

  # Merge original and mirrored datasets
  python space_mirror.py merge --src-paths /path/to/original /path/to/mirror --tgt-path /path/to/merged --repo-id my_dataset

  # Full pipeline (create mirror and merge)
  python space_mirror.py full --src-path /path/to/original --mirror-path /path/to/mirror --merge-path /path/to/merged --repo-id my_dataset --negate-joints 1 4 6
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # create-mirror command
    parser_create = subparsers.add_parser("create-mirror", help="Create mirrored dataset")
    parser_create.add_argument("--src-path", required=True, help="Source dataset path")
    parser_create.add_argument("--tgt-path", required=True, help="Target mirrored dataset path")
    _add_mirror_arguments(parser_create)

    # merge command
    parser_merge = subparsers.add_parser("merge", help="Merge datasets")
    parser_merge.add_argument("--src-paths", nargs="+", required=True, help="Source dataset paths list")
    parser_merge.add_argument("--tgt-path", required=True, help="Target merged dataset path")
    parser_merge.add_argument("--repo-id", required=True, help="Dataset repo_id")
    parser_merge.add_argument("--fps", type=int, default=30, help="FPS (default: 30)")
    parser_merge.add_argument("--robot-type", type=str, default="agilex", help="Robot type (default: agilex)")
    parser_merge.add_argument("--features-json", type=str, default=None, help="Path to features.json file")
    parser_merge.add_argument("--force", action="store_true", help="Force merge (ignore conflicts)")

    # full command
    parser_full = subparsers.add_parser("full", help="Full pipeline: create mirror and merge")
    parser_full.add_argument("--src-path", required=True, help="Source dataset path")
    parser_full.add_argument("--mirror-path", required=True, help="Mirrored dataset path")
    parser_full.add_argument("--merge-path", required=True, help="Merged dataset path")
    parser_full.add_argument("--repo-id", required=True, help="Dataset repo_id")
    _add_mirror_arguments(parser_full)
    parser_full.add_argument("--fps", type=int, default=30, help="FPS (default: 30)")
    parser_full.add_argument("--robot-type", type=str, default="agilex", help="Robot type (default: agilex)")
    parser_full.add_argument("--features-json", type=str, default=None, help="Path to features.json file")
    parser_full.add_argument("--force", action="store_true", help="Force merge (ignore conflicts)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "create-mirror":
        create_mirror_dataset(
            src_path=args.src_path,
            tgt_path=args.tgt_path,
            left_dim=args.left_dim,
            right_dim=args.right_dim,
            num_workers=args.num_workers,
            state_key=args.state_key,
            action_key=args.action_key,
            negate_joints=args.negate_joints,
            video_config=video_encode_config_from_args(args),
        )

    elif args.command == "merge":
        if not MERGE_AVAILABLE:
            print("Error: merge_lerobot module not available. Please ensure merge_lerobot.py is accessible.")
            sys.exit(1)

        features = None
        if args.features_json:
            with open(args.features_json, 'r', encoding='utf-8') as f:
                features = json.load(f)

        merge_lerobot_datasets(
            args.src_paths,
            args.tgt_path,
            args.repo_id,
            args.fps,
            args.robot_type,
            features,
            args.force
        )

    elif args.command == "full":
        print("=" * 60)
        print("Space Mirror Full Pipeline")
        print("=" * 60)
        print()

        # Step 1: Create mirrored dataset
        print("Step 1/2: Creating mirrored dataset")
        create_mirror_dataset(
            src_path=args.src_path,
            tgt_path=args.mirror_path,
            left_dim=args.left_dim,
            right_dim=args.right_dim,
            num_workers=args.num_workers,
            state_key=args.state_key,
            action_key=args.action_key,
            negate_joints=args.negate_joints,
            video_config=video_encode_config_from_args(args),
            merge_path=args.merge_path,
        )
        print()

        # Step 2: Merge datasets
        print("Step 2/2: Merging datasets")
        if not MERGE_AVAILABLE:
            print("Error: merge_lerobot module not available. Please ensure merge_lerobot.py is accessible.")
            sys.exit(1)

        features = None
        if args.features_json:
            with open(args.features_json, 'r', encoding='utf-8') as f:
                features = json.load(f)

        merge_lerobot_datasets(
            [args.src_path, args.mirror_path],
            args.merge_path,
            args.repo_id,
            args.fps,
            args.robot_type,
            features,
            args.force
        )
        print()
        print("=" * 60)
        print("✓ All processing complete!")
        print("=" * 60)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
