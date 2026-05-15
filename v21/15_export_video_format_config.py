#!/usr/bin/env python3
"""导出 LeRobot 数据集的视频编码参数配置。

该脚本会读取数据集 videos/ 下的 mp4，探测 codec、pix_fmt、fps、GOP、
B 帧、profile、level、bitrate 等视频相关参数，并写入 JSON 配置文件。
后续生成视频的工具可以读取该配置，按指定参数重编码。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from video_format_utils import (
    ffmpeg_codec_detail_args_for_source,
    ffmpeg_encoder_for_source,
    ffmpeg_gop_args_for_source,
    ffmpeg_profile_args_for_source,
    ffmpeg_rate_control_args_for_source,
    probe_video_format,
    video_fps,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SUMMARY_FIELDS = [
    "codec_name",
    "codec_tag_string",
    "profile",
    "pix_fmt",
    "width",
    "height",
    "bit_rate",
    "nb_frames",
    "nb_read_frames",
    "avg_frame_rate",
    "r_frame_rate",
    "fps",
    "has_b_frames",
    "level",
    "refs",
    "gop",
    "keyframes",
    "max_keyframe_gap_s",
    "mean_keyframe_gap_s",
    "format_name",
]

STRICT_CONSISTENCY_FIELDS = [
    "codec_name",
    "codec_tag_string",
    "profile",
    "pix_fmt",
    "width",
    "height",
    "avg_frame_rate",
    "r_frame_rate",
    "fps",
    "has_b_frames",
    "level",
    "refs",
    "format_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="探测 LeRobot 数据集视频参数，并导出可复用的 JSON 配置。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        "--dataset_dir",
        type=Path,
        required=True,
        help="LeRobot 数据集目录。",
    )
    parser.add_argument(
        "--output-json",
        "--output_json",
        type=Path,
        default=None,
        help="输出 JSON 配置路径；默认写到当前目录的 <dataset_name>_video_format_config.json。",
    )
    parser.add_argument(
        "--max-videos-per-key",
        "--max_videos_per_key",
        type=int,
        default=10,
        help="每个 video key 最多探测多少个视频；设为 0 或负数表示全量探测。",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON 缩进空格数。",
    )
    return parser.parse_args()


def ensure_tools() -> None:
    missing = [name for name in ("ffprobe",) if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"缺少依赖工具: {', '.join(missing)}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"数据集目录不存在或不是目录: {dataset_dir}")
    if not (dataset_dir / "videos").is_dir():
        raise FileNotFoundError(f"数据集 videos/ 目录不存在: {dataset_dir / 'videos'}")
    return dataset_dir


def default_output_json(dataset_dir: Path) -> Path:
    return Path.cwd() / f"{dataset_dir.name}_video_format_config.json"


def get_video_keys(dataset_dir: Path, info: dict[str, Any] | None) -> list[str]:
    if info:
        features = info.get("features", {})
        keys = [key for key, value in features.items() if value.get("dtype") == "video"]
        if keys:
            return sorted(keys)

    videos_dir = dataset_dir / "videos"
    keys = set()
    for video_path in videos_dir.rglob("*.mp4"):
        if video_path.parent.parent.name.startswith("chunk-"):
            keys.add(video_path.parent.name)
    return sorted(keys)


def find_videos_for_key(dataset_dir: Path, video_key: str) -> list[Path]:
    videos_dir = dataset_dir / "videos"
    paths: list[Path] = []
    for chunk_dir in sorted(videos_dir.glob("chunk-*")):
        key_dir = chunk_dir / video_key
        if key_dir.is_dir():
            paths.extend(sorted(key_dir.rglob("*.mp4")))

    if paths:
        return sorted(set(paths))

    # 兜底：兼容非标准目录结构。
    return sorted(
        path
        for path in videos_dir.rglob("*.mp4")
        if video_key in path.parts
    )


def evenly_sample_paths(paths: list[Path], max_count: int) -> list[Path]:
    if max_count <= 0 or len(paths) <= max_count:
        return paths
    if max_count == 1:
        return [paths[0]]

    last_index = len(paths) - 1
    indices = sorted({round(i * last_index / (max_count - 1)) for i in range(max_count)})
    return [paths[index] for index in indices]


def video_format_to_dict(fmt, dataset_dir: Path) -> dict[str, Any]:
    rel_path = fmt.path.relative_to(dataset_dir).as_posix() if fmt.path.is_relative_to(dataset_dir) else str(fmt.path)
    return {
        "path": rel_path,
        "codec_name": fmt.codec_name,
        "codec_tag_string": fmt.codec_tag_string,
        "profile": fmt.profile,
        "pix_fmt": fmt.pix_fmt,
        "width": fmt.width,
        "height": fmt.height,
        "has_b_frames": fmt.has_b_frames,
        "level": fmt.level,
        "refs": fmt.refs,
        "bit_rate": fmt.bit_rate,
        "nb_frames": fmt.nb_frames,
        "nb_read_frames": fmt.nb_read_frames,
        "r_frame_rate": fmt.r_frame_rate,
        "avg_frame_rate": fmt.avg_frame_rate,
        "fps": video_fps(fmt),
        "format_name": fmt.format_name,
        "keyframes": fmt.keyframes,
        "gop": fmt.gop,
        "max_keyframe_gap_s": fmt.max_keyframe_gap_s,
        "mean_keyframe_gap_s": fmt.mean_keyframe_gap_s,
    }


def summarize_values(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in SUMMARY_FIELDS:
        values = [sample.get(field) for sample in samples]
        unique_values = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        summary[field] = {
            "representative": unique_values[0] if unique_values else None,
            "unique": unique_values,
            "consistent": len(unique_values) <= 1,
        }

    numeric_fields = [
        "bit_rate",
        "nb_frames",
        "nb_read_frames",
        "fps",
        "keyframes",
        "gop",
        "max_keyframe_gap_s",
        "mean_keyframe_gap_s",
    ]
    for field in numeric_fields:
        values = [sample.get(field) for sample in samples if sample.get(field) is not None]
        if values:
            summary.setdefault(field, {"representative": values[0], "unique": [], "consistent": True})
            summary[field]["min"] = min(values)
            summary[field]["max"] = max(values)
            summary[field]["mean"] = sum(values) / len(values)
    return summary


def build_encoding_params(sample: dict[str, Any], fmt) -> dict[str, Any]:
    fps_arg = sample["avg_frame_rate"] or sample["r_frame_rate"] or str(sample["fps"])
    ffmpeg_args = (
        ["-c:v", ffmpeg_encoder_for_source(fmt)]
        + ffmpeg_profile_args_for_source(fmt)
        + ffmpeg_codec_detail_args_for_source(fmt)
        + ffmpeg_gop_args_for_source(fmt)
        + ffmpeg_rate_control_args_for_source(fmt)
        + ["-pix_fmt", sample["pix_fmt"] or "yuv420p", "-r", fps_arg, "-movflags", "+faststart"]
    )

    return {
        "codec_name": sample["codec_name"],
        "codec_tag_string": sample["codec_tag_string"],
        "ffmpeg_encoder": ffmpeg_encoder_for_source(fmt),
        "profile": sample["profile"],
        "pix_fmt": sample["pix_fmt"],
        "width": sample["width"],
        "height": sample["height"],
        "fps": sample["fps"],
        "fps_rate": sample["avg_frame_rate"] or sample["r_frame_rate"],
        "gop": sample["gop"],
        "keyint_min": sample["gop"],
        "sc_threshold": 0 if sample["gop"] is not None else None,
        "has_b_frames": sample["has_b_frames"],
        "bf": sample["has_b_frames"],
        "level": sample["level"],
        "refs": sample["refs"],
        "bit_rate": sample["bit_rate"],
        "movflags": "+faststart",
        "ffmpeg_output_args": ffmpeg_args,
    }


def probe_video_key(dataset_dir: Path, video_key: str, max_videos_per_key: int) -> dict[str, Any]:
    all_paths = find_videos_for_key(dataset_dir, video_key)
    sample_paths = evenly_sample_paths(all_paths, max_videos_per_key)

    if not sample_paths:
        return {
            "total_found": 0,
            "sample_count": 0,
            "representative": None,
            "encoding_params": None,
            "summary": {},
            "samples": [],
            "warnings": [f"未找到 video key 对应 mp4: {video_key}"],
        }

    logger.info("探测 video_key=%s: samples=%d/%d", video_key, len(sample_paths), len(all_paths))

    samples = []
    representative_fmt = None
    for path in sample_paths:
        fmt = probe_video_format(path)
        if representative_fmt is None:
            representative_fmt = fmt
        samples.append(video_format_to_dict(fmt, dataset_dir))

    assert representative_fmt is not None
    representative = samples[0]
    summary = summarize_values(samples)
    inconsistent_fields = [
        field
        for field in STRICT_CONSISTENCY_FIELDS
        if field in summary and not summary[field]["consistent"]
    ]

    return {
        "total_found": len(all_paths),
        "sample_count": len(samples),
        "representative": representative,
        "encoding_params": build_encoding_params(representative, representative_fmt),
        "summary": summary,
        "inconsistent_fields": inconsistent_fields,
        "samples": samples,
        "warnings": [] if not inconsistent_fields else [f"样本中这些字段不一致: {', '.join(inconsistent_fields)}"],
    }


def build_config(dataset_dir: Path, max_videos_per_key: int) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    info = read_json(info_path) if info_path.is_file() else None
    video_keys = get_video_keys(dataset_dir, info)
    if not video_keys:
        raise RuntimeError(f"未能找到视频 key: {dataset_dir}")

    key_configs = {
        video_key: probe_video_key(dataset_dir, video_key, max_videos_per_key)
        for video_key in video_keys
    }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "dataset_info": {
            "codebase_version": info.get("codebase_version") if info else None,
            "robot_type": info.get("robot_type") if info else None,
            "fps": info.get("fps") if info else None,
            "video_path": info.get("video_path") if info else None,
            "total_episodes": info.get("total_episodes") if info else None,
            "total_frames": info.get("total_frames") if info else None,
        },
        "sampling": {
            "max_videos_per_key": max_videos_per_key,
            "mode": "all" if max_videos_per_key <= 0 else "evenly_sampled",
        },
        "video_keys": key_configs,
    }


def print_report(config: dict[str, Any]) -> None:
    print("=" * 80)
    print("LeRobot 视频参数配置")
    print("=" * 80)
    print(f"dataset_dir: {config['dataset_dir']}")
    print(f"sampling: {config['sampling']}")
    print()

    for video_key, key_config in config["video_keys"].items():
        print(f"[{video_key}]")
        print(f"  videos: samples={key_config['sample_count']}/{key_config['total_found']}")
        representative = key_config.get("representative")
        if representative:
            print(
                "  params: "
                f"codec={representative['codec_name']}, "
                f"tag={representative['codec_tag_string']}, "
                f"profile={representative['profile']}, "
                f"pix_fmt={representative['pix_fmt']}, "
                f"size={representative['width']}x{representative['height']}, "
                f"fps={representative['avg_frame_rate'] or representative['r_frame_rate']}, "
                f"frames={representative['nb_read_frames'] or representative['nb_frames']}, "
                f"gop~{representative['gop']}, "
                f"keyframes={representative['keyframes']}, "
                f"has_b_frames={representative['has_b_frames']}, "
                f"level={representative['level']}, "
                f"refs={representative['refs']}, "
                f"bit_rate={representative['bit_rate']}"
            )
            print(f"  representative: {representative['path']}")
        for warning in key_config.get("warnings", []):
            print(f"  warning: {warning}")
        print()


def main() -> int:
    args = parse_args()
    ensure_tools()
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    output_json = args.output_json.expanduser().resolve() if args.output_json else default_output_json(dataset_dir)

    config = build_config(dataset_dir, args.max_videos_per_key)
    print_report(config)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=args.indent, ensure_ascii=False)
        f.write("\n")

    logger.info("已写入视频参数配置: %s", output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
