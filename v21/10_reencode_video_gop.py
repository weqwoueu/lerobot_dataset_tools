#!/usr/bin/env python3
"""重新编码 LeRobot 数据集 mp4 视频的 GOP，并支持只检测关键帧间隔。

默认模式会把整个数据集复制到新的输出目录，然后将输出目录中的旧 mp4
移动到 old_video/，再把重编码后的 mp4 写回 videos/。源数据集不会被修改。

示例:
  # 只检测当前数据集视频关键帧间隔
  python tools/lerobot_dataset_tools/10_reencode_video_gop.py \
      --dataset_dir .cache/huggingface/lerobot/standard/darwin02_0503_trim_tail_force_delta \
      --check

  # 非原地重编码，默认 gop=2
  python tools/lerobot_dataset_tools/10_reencode_video_gop.py \
      --dataset_dir .cache/huggingface/lerobot/standard/darwin02_0503_trim_tail_force_delta \
      --output_dir .cache/huggingface/lerobot/standard/darwin02_0503_trim_tail_force_delta_gop2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GopStats:
    video_path: Path
    keyframes: int
    max_gap_s: float | None
    mean_gap_s: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="非原地重新编码 LeRobot 数据集 videos/ 下的 mp4 GOP，或检测关键帧间隔。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="输入 LeRobot 数据集目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="输出数据集目录。--check 模式不需要；转换模式必须提供，且目录不能已存在。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检测 videos/ 下 mp4 的关键帧间隔，不复制或重编码数据集。",
    )
    parser.add_argument(
        "--gop",
        type=int,
        default=2,
        help="ffmpeg -g 参数，即关键帧间隔。默认 2，适合 LeRobot 训练随机读帧。",
    )
    parser.add_argument(
        "--codec",
        default="libx264",
        help="ffmpeg 视频编码器。",
    )
    parser.add_argument(
        "--pix_fmt",
        default="yuv420p",
        help="ffmpeg 像素格式。",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="可选 CRF 质量参数。未指定时使用 ffmpeg/libx264 默认值。",
    )
    parser.add_argument(
        "--ffmpeg_loglevel",
        default="error",
        help="ffmpeg 日志级别。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="转换模式下并行运行的 ffmpeg 进程数。",
    )
    parser.add_argument(
        "--max_check_videos",
        type=int,
        default=None,
        help="检测模式最多检查多少个视频；未指定则检查全部。",
    )
    parser.add_argument(
        "--warn_gap_s",
        type=float,
        default=0.5,
        help="检测模式中，最大关键帧间隔超过该秒数会计为告警。",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="转换模式下只打印计划，不复制或重编码。",
    )
    return parser.parse_args()


def ensure_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"缺少依赖工具: {', '.join(missing)}")


def resolve_dataset_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"路径不是目录: {dataset_dir}")
    if not (dataset_dir / "videos").is_dir():
        raise FileNotFoundError(f"数据集 videos/ 目录不存在: {dataset_dir / 'videos'}")
    return dataset_dir


def iter_video_files(videos_dir: Path) -> list[Path]:
    return sorted(videos_dir.rglob("*.mp4"))


def probe_video_frame_count(video_path: Path) -> int:
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
        raise RuntimeError(f"无法读取视频帧数: {video_path}")
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
    timestamps = []
    for line in result.stdout.splitlines():
        value = line.strip().split(",", 1)[0]
        if value and value != "N/A":
            timestamps.append(float(value))
    return timestamps


def compute_gop_stats(video_path: Path) -> GopStats:
    timestamps = probe_keyframe_timestamps(video_path)
    if len(timestamps) < 2:
        return GopStats(video_path=video_path, keyframes=len(timestamps), max_gap_s=None, mean_gap_s=None)
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    return GopStats(
        video_path=video_path,
        keyframes=len(timestamps),
        max_gap_s=max(gaps),
        mean_gap_s=sum(gaps) / len(gaps),
    )


def check_gop(dataset_dir: Path, max_videos: int | None, warn_gap_s: float) -> None:
    videos = iter_video_files(dataset_dir / "videos")
    if max_videos is not None:
        videos = videos[:max_videos]
    if not videos:
        raise FileNotFoundError(f"未找到 mp4: {dataset_dir / 'videos'}")

    logger.info("开始检测 GOP: videos=%d, warn_gap_s=%.3f", len(videos), warn_gap_s)
    stats = [compute_gop_stats(video_path) for video_path in videos]
    warned = [
        item
        for item in stats
        if item.max_gap_s is None or item.max_gap_s > warn_gap_s
    ]

    gaps = [item.max_gap_s for item in stats if item.max_gap_s is not None]
    if gaps:
        logger.info(
            "最大关键帧间隔: min=%.3fs, mean=%.3fs, max=%.3fs",
            min(gaps),
            sum(gaps) / len(gaps),
            max(gaps),
        )
    logger.info("视频数: %d, 告警数: %d", len(stats), len(warned))

    for item in sorted(warned, key=lambda x: x.max_gap_s or -1, reverse=True)[:20]:
        rel_path = item.video_path.relative_to(dataset_dir)
        if item.max_gap_s is None:
            logger.warning("%s: keyframes=%d, 无法计算关键帧间隔", rel_path, item.keyframes)
        else:
            logger.warning(
                "%s: keyframes=%d, mean_gap=%.3fs, max_gap=%.3fs",
                rel_path,
                item.keyframes,
                item.mean_gap_s or 0.0,
                item.max_gap_s,
            )


def build_ffmpeg_cmd(
    src_video_path: Path,
    dst_video_path: Path,
    gop: int,
    codec: str,
    pix_fmt: str,
    crf: int | None,
    ffmpeg_loglevel: str,
) -> list[str]:
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
        "-c:v",
        codec,
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        pix_fmt,
        "-movflags",
        "+faststart",
    ]
    if crf is not None:
        cmd.extend(["-crf", str(crf)])
    cmd.append(str(dst_video_path))
    return cmd


def reencode_one(
    old_video_path: Path,
    new_video_path: Path,
    gop: int,
    codec: str,
    pix_fmt: str,
    crf: int | None,
    ffmpeg_loglevel: str,
) -> None:
    new_video_path.parent.mkdir(parents=True, exist_ok=True)
    old_frames = probe_video_frame_count(old_video_path)
    with tempfile.NamedTemporaryFile(
        prefix=f"{new_video_path.stem}.gop_",
        suffix=".mp4",
        dir=new_video_path.parent,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        tmp_path.unlink()
        cmd = build_ffmpeg_cmd(
            src_video_path=old_video_path,
            dst_video_path=tmp_path,
            gop=gop,
            codec=codec,
            pix_fmt=pix_fmt,
            crf=crf,
            ffmpeg_loglevel=ffmpeg_loglevel,
        )
        subprocess.run(cmd, check=True)
        new_frames = probe_video_frame_count(tmp_path)
        if new_frames != old_frames:
            raise RuntimeError(
                f"重编码后帧数不一致: {old_video_path}, 原 {old_frames}, 新 {new_frames}"
            )
        tmp_path.replace(new_video_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def copy_dataset(src_dataset_dir: Path, output_dir: Path, dry_run: bool) -> None:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，请换一个路径: {output_dir}")
    logger.info("复制数据集: %s -> %s", src_dataset_dir, output_dir)
    if not dry_run:
        shutil.copytree(src_dataset_dir, output_dir)


def move_old_videos(output_dir: Path, dry_run: bool) -> tuple[Path, list[Path]]:
    videos_dir = output_dir / "videos"
    old_videos_dir = output_dir / "old_video"
    video_files = iter_video_files(videos_dir)
    if not video_files:
        raise FileNotFoundError(f"未找到 mp4: {videos_dir}")

    logger.info("移动旧视频到备份目录: %s, files=%d", old_videos_dir, len(video_files))
    if dry_run:
        return old_videos_dir, [old_videos_dir / item.relative_to(videos_dir) for item in video_files]

    old_videos_dir.mkdir(parents=True)
    moved_paths = []
    for video_path in video_files:
        rel_path = video_path.relative_to(videos_dir)
        old_video_path = old_videos_dir / rel_path
        old_video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(video_path), str(old_video_path))
        moved_paths.append(old_video_path)
    return old_videos_dir, moved_paths


def reencode_dataset(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        raise ValueError("转换模式必须指定 --output_dir")
    output_dir = args.output_dir.resolve()
    if args.gop <= 0:
        raise ValueError(f"--gop 必须大于 0，当前为 {args.gop}")
    if args.workers <= 0:
        raise ValueError(f"--workers 必须大于 0，当前为 {args.workers}")

    if args.dry_run:
        src_videos_dir = args.dataset_dir / "videos"
        old_videos_dir = output_dir / "old_video"
        video_paths = iter_video_files(src_videos_dir)
        logger.info("[dry-run] 将复制数据集: %s -> %s", args.dataset_dir, output_dir)
        logger.info("[dry-run] 将移动旧视频到: %s", old_videos_dir)
        for video_path in video_paths[:20]:
            rel_path = video_path.relative_to(src_videos_dir)
            logger.info("[dry-run] %s -> %s", old_videos_dir / rel_path, output_dir / "videos" / rel_path)
        if len(video_paths) > 20:
            logger.info("[dry-run] 另有 %d 个视频未列出", len(video_paths) - 20)
        return

    copy_dataset(args.dataset_dir, output_dir, args.dry_run)
    videos_dir = output_dir / "videos"
    old_videos_dir, old_video_paths = move_old_videos(output_dir, args.dry_run)

    logger.info(
        "开始重编码: files=%d, gop=%d, codec=%s, pix_fmt=%s",
        len(old_video_paths),
        args.gop,
        args.codec,
        args.pix_fmt,
    )

    def submit_one(old_video_path: Path) -> None:
        rel_path = old_video_path.relative_to(old_videos_dir)
        reencode_one(
            old_video_path=old_video_path,
            new_video_path=videos_dir / rel_path,
            gop=args.gop,
            codec=args.codec,
            pix_fmt=args.pix_fmt,
            crf=args.crf,
            ffmpeg_loglevel=args.ffmpeg_loglevel,
        )

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(submit_one, path) for path in old_video_paths]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 20 == 0 or completed == len(old_video_paths):
                logger.info("  已重编码 %d/%d", completed, len(old_video_paths))

    logger.info("完成: %s", output_dir)
    logger.info("旧视频备份目录: %s", output_dir / "old_video")


def main() -> int:
    args = parse_args()
    ensure_tools()
    args.dataset_dir = resolve_dataset_dir(args.dataset_dir)

    if args.check:
        check_gop(args.dataset_dir, args.max_check_videos, args.warn_gap_s)
        return 0

    reencode_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
