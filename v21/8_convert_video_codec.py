#!/usr/bin/env python3
"""批量转换 LeRobot 数据集 videos/ 下 mp4 文件的视频编码。

支持两种目标编码：
1. av1   : 使用 ffmpeg 的 libaom-av1 编码器
2. mp4v  : 使用 MPEG-4 Part 2，并写入 mp4v tag

脚本会先用 ffprobe 检查当前视频编码，只转换符合条件的文件，并以临时文件完成原地替换。

示例：
    # 将数据集 videos/ 下所有 av1 视频转成 mp4v
    python tools/lerobot_dataset_tools/6_convert_video_codec.py \
        --dataset_dir .cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human \
        --target_codec mp4v \
        --source_codec av1

    # 将 videos/ 下所有 mp4v 视频转回 av1
    python tools/lerobot_dataset_tools/6_convert_video_codec.py \
        --videos_dir .cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human/videos \
        --target_codec av1 \
        --source_codec mp4v
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SUPPORTED_CODECS = {"av1", "mp4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量转换 LeRobot 数据集 mp4 视频编码")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset_dir",
        type=Path,
        help="LeRobot 数据集目录，脚本会自动使用其中的 videos/ 子目录",
    )
    group.add_argument(
        "--videos_dir",
        type=Path,
        help="直接指定 videos/ 目录",
    )
    parser.add_argument(
        "--target_codec",
        choices=sorted(SUPPORTED_CODECS),
        required=True,
        help="目标编码，只支持 av1 或 mp4v",
    )
    parser.add_argument(
        "--source_codec",
        choices=sorted(SUPPORTED_CODECS),
        default=None,
        help="仅转换当前编码匹配该值的文件；未指定时，转换所有非目标编码文件",
    )
    parser.add_argument(
        "--ffmpeg_loglevel",
        default="error",
        help="ffmpeg 日志级别，默认 error",
    )
    parser.add_argument(
        "--mp4v_qscale",
        type=int,
        default=3,
        help="转换到 mp4v 时使用的 qscale，默认 3",
    )
    parser.add_argument(
        "--av1_crf",
        type=int,
        default=30,
        help="转换到 av1 时使用的 CRF，默认 30",
    )
    parser.add_argument(
        "--av1_cpu_used",
        type=int,
        default=8,
        help="转换到 av1 时 libaom-av1 的 cpu-used 参数，默认 8",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印将要转换的文件，不实际执行 ffmpeg",
    )
    return parser.parse_args()


def ensure_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"缺少依赖工具: {', '.join(missing)}")


def resolve_videos_dir(args: argparse.Namespace) -> Path:
    videos_dir = args.videos_dir if args.videos_dir is not None else args.dataset_dir / "videos"
    videos_dir = videos_dir.resolve()
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos 目录不存在: {videos_dir}")
    if not videos_dir.is_dir():
        raise NotADirectoryError(f"路径不是目录: {videos_dir}")
    return videos_dir


def iter_video_files(videos_dir: Path) -> list[Path]:
    return sorted(videos_dir.rglob("*.mp4"))


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


def probe_codec(video_path: Path) -> str:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,codec_tag_string",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    codec_name = values.get("codec_name", "")
    codec_tag = values.get("codec_tag_string", "")
    if not codec_name:
        raise RuntimeError(f"无法识别视频编码: {video_path}")
    return canonicalize_codec(codec_name, codec_tag)


def build_ffmpeg_cmd(
    input_path: Path,
    output_path: Path,
    target_codec: str,
    ffmpeg_loglevel: str,
    mp4v_qscale: int,
    av1_crf: int,
    av1_cpu_used: int,
) -> list[str]:
    common_prefix = [
        "ffmpeg",
        "-y",
        "-loglevel",
        ffmpeg_loglevel,
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-pix_fmt",
        "yuv420p",
        "-an",
    ]

    if target_codec == "mp4v":
        return common_prefix + [
            "-c:v",
            "mpeg4",
            "-vtag",
            "mp4v",
            "-q:v",
            str(mp4v_qscale),
            str(output_path),
        ]

    if target_codec == "av1":
        return common_prefix + [
            "-c:v",
            "libaom-av1",
            "-crf",
            str(av1_crf),
            "-b:v",
            "0",
            "-cpu-used",
            str(av1_cpu_used),
            "-row-mt",
            "1",
            str(output_path),
        ]

    raise ValueError(f"不支持的目标编码: {target_codec}")


def should_convert(current_codec: str, target_codec: str, source_codec: str | None) -> bool:
    if current_codec == target_codec:
        return False
    if source_codec is not None and current_codec != source_codec:
        return False
    return True


def convert_one(
    video_path: Path,
    current_codec: str,
    args: argparse.Namespace,
) -> bool:
    rel_name = str(video_path)
    logger.info("转换 %s: %s -> %s", rel_name, current_codec, args.target_codec)
    if args.dry_run:
        return True

    tmp_path = video_path.with_name(f"{video_path.stem}.tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()

    cmd = build_ffmpeg_cmd(
        input_path=video_path,
        output_path=tmp_path,
        target_codec=args.target_codec,
        ffmpeg_loglevel=args.ffmpeg_loglevel,
        mp4v_qscale=args.mp4v_qscale,
        av1_crf=args.av1_crf,
        av1_cpu_used=args.av1_cpu_used,
    )

    try:
        subprocess.run(cmd, check=True)
        new_codec = probe_codec(tmp_path)
        if new_codec != args.target_codec:
            raise RuntimeError(
                f"转码后编码不匹配: {video_path} -> {tmp_path}, "
                f"期望 {args.target_codec}, 实际 {new_codec}"
            )
        tmp_path.replace(video_path)
        return True
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main() -> int:
    args = parse_args()
    ensure_tools()
    videos_dir = resolve_videos_dir(args)
    video_files = iter_video_files(videos_dir)

    if not video_files:
        logger.warning("未找到 mp4 文件: %s", videos_dir)
        return 0

    logger.info("videos 目录: %s", videos_dir)
    logger.info("共找到 %d 个 mp4 文件", len(video_files))
    if args.source_codec is not None:
        logger.info("仅转换当前编码为 %s 的文件", args.source_codec)

    converted = 0
    skipped = 0

    for video_path in video_files:
        current_codec = probe_codec(video_path)
        if not should_convert(current_codec, args.target_codec, args.source_codec):
            skipped += 1
            continue
        convert_one(video_path, current_codec, args)
        converted += 1

    logger.info("完成: converted=%d skipped=%d total=%d", converted, skipped, len(video_files))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.error("用户中断")
        raise SystemExit(130)
    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1)
