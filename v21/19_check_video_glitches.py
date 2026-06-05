#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测 LeRobot v2.1 数据集中 mp4 是否存在疑似画面撕裂/错位异常。

脚本只读取数据集，不修改任何文件。它会根据 meta/info.json 中声明的
video feature 和 video_path 模板，遍历每个 episode 的每个摄像头 mp4，
并按 episode 汇总打印异常结果。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoTask:
    episode_index: int
    camera: str
    video_path: Path


@dataclass(frozen=True)
class VideoResult:
    episode_index: int
    camera: str
    video_path: Path
    checked_frames: int
    bad_frames: int
    strong_bad_frames: int
    max_score: float
    error: str | None = None

    @property
    def is_decode_error(self) -> bool:
        return self.error is not None

    @property
    def bad_ratio(self) -> float:
        if self.checked_frames <= 0:
            return 0.0
        return self.bad_frames / self.checked_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检测 LeRobot v2.1 数据集 mp4 中疑似画面撕裂/错位的 episode。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        "--dataset-dir",
        type=Path,
        required=True,
        help="LeRobot v2.1 数据集目录。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行检测的视频数量。",
    )
    parser.add_argument(
        "--sample_stride",
        "--sample-stride",
        type=int,
        default=1,
        help="每隔多少帧检测一次；1 表示逐帧检测。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="异常帧分数阈值。",
    )
    parser.add_argument(
        "--min_bad_frames",
        "--min-bad-frames",
        type=int,
        default=2,
        help="同一个 mp4 至少命中多少个异常帧才判为异常。",
    )
    parser.add_argument(
        "--min_bad_ratio",
        "--min-bad-ratio",
        type=float,
        default=0.03,
        help="同一个 mp4 的异常帧占比至少达到多少才判为异常，用于过滤零星误报。",
    )
    parser.add_argument(
        "--strong_threshold",
        "--strong-threshold",
        type=float,
        default=0.70,
        help="强异常帧分数阈值；强异常帧可绕过异常帧占比要求。",
    )
    parser.add_argument(
        "--min_strong_bad_frames",
        "--min-strong-bad-frames",
        type=int,
        default=2,
        help="同一个 mp4 至少命中多少个强异常帧才判为异常。",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="只检测指定 camera/video feature；不传则检测所有 video feature。",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="只检测指定 episode，支持逗号和范围，例如 55,56 或 50-60。",
    )
    parser.add_argument(
        "--max_episodes",
        "--max-episodes",
        type=int,
        default=0,
        help="最多检测多少个 episode；0 表示全量。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise ValueError(f"--workers 必须大于 0，当前为 {args.workers}")
    if args.sample_stride <= 0:
        raise ValueError(f"--sample_stride 必须大于 0，当前为 {args.sample_stride}")
    if args.min_bad_frames <= 0:
        raise ValueError(f"--min_bad_frames 必须大于 0，当前为 {args.min_bad_frames}")
    if args.min_bad_ratio < 0:
        raise ValueError(f"--min_bad_ratio 不能小于 0，当前为 {args.min_bad_ratio}")
    if args.min_bad_ratio > 1:
        raise ValueError(f"--min_bad_ratio 不能大于 1，当前为 {args.min_bad_ratio}")
    if args.strong_threshold <= 0:
        raise ValueError(f"--strong_threshold 必须大于 0，当前为 {args.strong_threshold}")
    if args.strong_threshold < args.threshold:
        raise ValueError("--strong_threshold 必须大于或等于 --threshold")
    if args.min_strong_bad_frames <= 0:
        raise ValueError(f"--min_strong_bad_frames 必须大于 0，当前为 {args.min_strong_bad_frames}")
    if args.max_episodes < 0:
        raise ValueError(f"--max_episodes 不能小于 0，当前为 {args.max_episodes}")
    if args.threshold <= 0:
        raise ValueError(f"--threshold 必须大于 0，当前为 {args.threshold}")
    if args.episodes is not None and args.max_episodes > 0:
        raise ValueError("--episodes 和 --max_episodes 不能同时使用")


def read_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"未找到 meta/info.json: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"数据集目录不存在或不是目录: {dataset_dir}")
    if not (dataset_dir / "videos").is_dir():
        raise FileNotFoundError(f"数据集 videos/ 目录不存在: {dataset_dir / 'videos'}")
    return dataset_dir


def get_video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json 中缺少 features 字段或格式不正确")

    video_keys = [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if not video_keys:
        raise ValueError("meta/info.json 的 features 中没有 dtype == 'video' 的字段")
    return sorted(video_keys)


def require_positive_int(info: dict[str, Any], key: str) -> int:
    value = info.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"meta/info.json 中 {key} 必须是正整数，当前为 {value!r}")
    return value


def parse_episode_selector(value: str, total_episodes: int) -> list[int]:
    episodes: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"episode 范围无效: {part}")
            episodes.update(range(start, end + 1))
        else:
            episodes.add(int(part))

    invalid = sorted(ep for ep in episodes if ep < 0 or ep >= total_episodes)
    if invalid:
        raise ValueError(f"--episodes 包含超出范围的 episode: {invalid}，有效范围是 0-{total_episodes - 1}")
    return sorted(episodes)


def build_video_path(
    dataset_dir: Path,
    video_path_template: str,
    video_key: str,
    episode_index: int,
    episode_chunk: int,
) -> Path:
    try:
        rel_path = video_path_template.format(
            video_key=video_key,
            episode_index=episode_index,
            episode_chunk=episode_chunk,
        )
    except KeyError as exc:
        raise ValueError(f"video_path 模板包含不支持的占位符: {exc}") from exc
    return dataset_dir / rel_path


def build_video_tasks(
    dataset_dir: Path,
    info: dict[str, Any],
    camera: str | None,
    episodes: str | None,
    max_episodes: int,
) -> list[VideoTask]:
    video_path_template = info.get("video_path")
    if not isinstance(video_path_template, str) or not video_path_template:
        raise ValueError("meta/info.json 中缺少有效的 video_path 模板")

    total_episodes = require_positive_int(info, "total_episodes")
    chunks_size = require_positive_int(info, "chunks_size")
    video_keys = get_video_keys(info)

    if camera is not None:
        if camera not in video_keys:
            available = ", ".join(video_keys)
            raise ValueError(f"指定的 --camera 不存在: {camera}；可用 camera: {available}")
        video_keys = [camera]

    if episodes is not None:
        episode_indices = parse_episode_selector(episodes, total_episodes)
    else:
        episode_count = total_episodes if max_episodes <= 0 else min(total_episodes, max_episodes)
        episode_indices = list(range(episode_count))

    tasks: list[VideoTask] = []
    for episode_index in episode_indices:
        episode_chunk = episode_index // chunks_size
        for video_key in video_keys:
            tasks.append(
                VideoTask(
                    episode_index=episode_index,
                    camera=video_key,
                    video_path=build_video_path(
                        dataset_dir=dataset_dir,
                        video_path_template=video_path_template,
                        video_key=video_key,
                        episode_index=episode_index,
                        episode_chunk=episode_chunk,
                    ),
                )
            )
    return tasks


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def preprocess_frame(frame: np.ndarray, max_width: int = 320) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_size = (max_width, max(8, int(round(height * scale))))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def preprocess_rgb_frame(frame: np.ndarray, max_width: int = 320) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_size = (max_width, max(8, int(round(height * scale))))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def robust_spike_score(values: np.ndarray) -> tuple[float, float, int]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-6
    p95 = float(np.percentile(values, 95))
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])

    robust_z = (peak - median) / (1.4826 * mad)
    z_score = clip01((robust_z - 6.0) / 18.0)
    peak_score = clip01((peak - p95) / 0.10)
    return z_score, peak_score, peak_index


def compute_frame_glitch_score(gray: np.ndarray, prev_gray: np.ndarray | None) -> float:
    """返回 0 到 1 之间的疑似撕裂分数。

    分数由两部分组成：
    1. 单帧内部是否存在横向、覆盖范围较宽、非常突兀的行边界；
    2. 该行边界或水平区域变化是否相对上一帧具有明显新颖性。
    """
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return 0.0

    row_abs = np.abs(np.diff(gray, axis=0))
    line_mean = row_abs.mean(axis=1)
    if line_mean.size == 0:
        return 0.0

    z_score, peak_score, peak_index = robust_spike_score(line_mean)
    peak_line = row_abs[peak_index]
    coverage = float(np.mean(peak_line > 0.12))
    coverage_score = clip01((coverage - 0.35) / 0.45)

    local_start = max(0, peak_index - 8)
    local_end = min(line_mean.size, peak_index + 9)
    local_values = np.delete(line_mean[local_start:local_end], peak_index - local_start)
    local_baseline = float(np.median(local_values)) if local_values.size else 0.0
    local_score = clip01((float(line_mean[peak_index]) - local_baseline) / 0.12)

    spatial_score = (
        0.35 * z_score
        + 0.30 * peak_score
        + 0.25 * coverage_score
        + 0.10 * local_score
    )

    if prev_gray is None or prev_gray.shape != gray.shape:
        return spatial_score * 0.45

    prev_row_abs = np.abs(np.diff(prev_gray, axis=0))
    if prev_row_abs.shape != row_abs.shape:
        return spatial_score * 0.45

    novelty_line = np.maximum(peak_line - prev_row_abs[peak_index], 0.0)
    novelty_score = clip01((float(novelty_line.mean()) - 0.04) / 0.12)

    frame_delta = np.abs(gray - prev_gray)
    row_motion = frame_delta.mean(axis=1)
    if row_motion.size >= 2:
        motion_step = float(np.max(np.abs(np.diff(row_motion))))
        motion_step_score = clip01((motion_step - 0.035) / 0.12)
    else:
        motion_step_score = 0.0

    row_motion_median = float(np.median(row_motion))
    row_motion_mad = float(np.median(np.abs(row_motion - row_motion_median))) + 1e-6
    motion_peak_row = min(peak_index, row_motion.size - 1)
    motion_robust_z = (float(row_motion[motion_peak_row]) - row_motion_median) / (1.4826 * row_motion_mad)
    motion_outlier_score = clip01((motion_robust_z - 5.0) / 15.0)

    temporal_score = max(novelty_score, motion_step_score, motion_outlier_score)
    combined_score = 0.60 * spatial_score + 0.40 * temporal_score

    if temporal_score < 0.15:
        combined_score *= 0.45
    return clip01(combined_score)


def expected_frame_count_pyav(video_path: Path) -> int:
    try:
        import av
    except ImportError:
        return 0

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            return int(stream.frames or 0)
    except Exception:
        return 0


def detect_video_glitches_with_pyav(
    task: VideoTask,
    sample_stride: int,
    threshold: float,
    strong_threshold: float,
) -> VideoResult | None:
    try:
        import av
    except ImportError:
        return None

    video_path = task.video_path
    expected_frames = 0
    frame_index = 0
    checked_frames = 0
    bad_frames = 0
    strong_bad_frames = 0
    max_score = 0.0
    prev_gray: np.ndarray | None = None

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            expected_frames = int(stream.frames or 0)
            for frame in container.decode(video=0):
                if frame_index % sample_stride == 0:
                    gray = preprocess_rgb_frame(frame.to_ndarray(format="rgb24"))
                    score = compute_frame_glitch_score(gray, prev_gray)
                    checked_frames += 1
                    max_score = max(max_score, score)
                    if score >= threshold:
                        bad_frames += 1
                    if score >= strong_threshold:
                        strong_bad_frames += 1
                    prev_gray = gray
                else:
                    prev_gray = preprocess_rgb_frame(frame.to_ndarray(format="rgb24"))
                frame_index += 1
    except Exception as exc:
        return VideoResult(
            episode_index=task.episode_index,
            camera=task.camera,
            video_path=video_path,
            checked_frames=checked_frames,
            bad_frames=bad_frames,
            strong_bad_frames=strong_bad_frames,
            max_score=max_score,
            error=f"PyAV 解码失败: {type(exc).__name__}: {exc}",
        )

    if frame_index == 0:
        return VideoResult(
            episode_index=task.episode_index,
            camera=task.camera,
            video_path=video_path,
            checked_frames=0,
            bad_frames=0,
            strong_bad_frames=0,
            max_score=0.0,
            error="无法读取首帧",
        )

    if expected_frames > 0 and frame_index + 2 < expected_frames:
        return VideoResult(
            episode_index=task.episode_index,
            camera=task.camera,
            video_path=video_path,
            checked_frames=checked_frames,
            bad_frames=bad_frames,
            strong_bad_frames=strong_bad_frames,
            max_score=max_score,
            error=f"解码帧数不足，期望约 {expected_frames} 帧，实际读取 {frame_index} 帧",
        )

    return VideoResult(
        episode_index=task.episode_index,
        camera=task.camera,
        video_path=video_path,
        checked_frames=checked_frames,
        bad_frames=bad_frames,
        strong_bad_frames=strong_bad_frames,
        max_score=max_score,
        error=None,
    )


def detect_video_glitches(
    task: VideoTask,
    sample_stride: int,
    threshold: float,
    strong_threshold: float,
) -> VideoResult:
    video_path = task.video_path
    if not video_path.exists():
        return VideoResult(
            episode_index=task.episode_index,
            camera=task.camera,
            video_path=video_path,
            checked_frames=0,
            bad_frames=0,
            strong_bad_frames=0,
            max_score=0.0,
            error="视频文件缺失",
        )
    if video_path.stat().st_size == 0:
        return VideoResult(
            episode_index=task.episode_index,
            camera=task.camera,
            video_path=video_path,
            checked_frames=0,
            bad_frames=0,
            strong_bad_frames=0,
            max_score=0.0,
            error="视频文件为空",
        )

    pyav_result = detect_video_glitches_with_pyav(task, sample_stride, threshold, strong_threshold)
    if pyav_result is not None and not pyav_result.is_decode_error:
        return pyav_result

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return pyav_result or VideoResult(
            episode_index=task.episode_index,
            camera=task.camera,
            video_path=video_path,
            checked_frames=0,
            bad_frames=0,
            strong_bad_frames=0,
            max_score=0.0,
            error="OpenCV 无法打开视频",
        )

    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0
    checked_frames = 0
    bad_frames = 0
    strong_bad_frames = 0
    max_score = 0.0
    prev_gray: np.ndarray | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = preprocess_frame(frame)
            if frame_index % sample_stride == 0:
                score = compute_frame_glitch_score(gray, prev_gray)
                checked_frames += 1
                max_score = max(max_score, score)
                if score >= threshold:
                    bad_frames += 1
                if score >= strong_threshold:
                    strong_bad_frames += 1

            prev_gray = gray
            frame_index += 1
    finally:
        capture.release()

    if frame_index == 0:
        return pyav_result or VideoResult(
            episode_index=task.episode_index,
            camera=task.camera,
            video_path=video_path,
            checked_frames=0,
            bad_frames=0,
            strong_bad_frames=0,
            max_score=0.0,
            error="无法读取首帧",
        )

    if expected_frames <= 0:
        expected_frames = expected_frame_count_pyav(video_path)

    if expected_frames > 0 and frame_index + 2 < expected_frames:
        return VideoResult(
            episode_index=task.episode_index,
            camera=task.camera,
            video_path=video_path,
            checked_frames=checked_frames,
            bad_frames=bad_frames,
            strong_bad_frames=strong_bad_frames,
            max_score=max_score,
            error=f"解码帧数不足，期望约 {expected_frames} 帧，实际读取 {frame_index} 帧",
        )

    return VideoResult(
        episode_index=task.episode_index,
        camera=task.camera,
        video_path=video_path,
        checked_frames=checked_frames,
        bad_frames=bad_frames,
        strong_bad_frames=strong_bad_frames,
        max_score=max_score,
        error=None,
    )


def relative_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def is_abnormal_result(
    result: VideoResult,
    min_bad_frames: int,
    min_bad_ratio: float,
    min_strong_bad_frames: int,
) -> bool:
    if result.is_decode_error:
        return True
    has_many_bad_frames = result.bad_frames >= min_bad_frames and result.bad_ratio >= min_bad_ratio
    has_strong_bad_frames = result.strong_bad_frames >= min_strong_bad_frames
    return has_many_bad_frames or has_strong_bad_frames


def print_summary(
    results: list[VideoResult],
    dataset_dir: Path,
    min_bad_frames: int,
    min_bad_ratio: float,
    min_strong_bad_frames: int,
) -> None:
    abnormal_results = [
        result
        for result in results
        if is_abnormal_result(
            result=result,
            min_bad_frames=min_bad_frames,
            min_bad_ratio=min_bad_ratio,
            min_strong_bad_frames=min_strong_bad_frames,
        )
    ]
    abnormal_episode_ids = sorted({result.episode_index for result in abnormal_results})

    print()
    print("=" * 80)
    print("LeRobot 视频异常检测结果")
    print("=" * 80)
    print(f"检测视频数: {len(results)}")
    print(f"异常视频数: {len(abnormal_results)}")
    print(f"异常 episode ids: {abnormal_episode_ids}")

    if not abnormal_results:
        print("未发现疑似异常视频。")
        return

    by_episode: dict[int, list[VideoResult]] = {}
    for result in abnormal_results:
        by_episode.setdefault(result.episode_index, []).append(result)

    for episode_index in abnormal_episode_ids:
        print(f"\nepisode {episode_index}:")
        episode_results = sorted(by_episode[episode_index], key=lambda item: item.camera)
        for result in episode_results:
            rel_path = relative_path(result.video_path, dataset_dir)
            error_text = f", error={result.error}" if result.error else ""
            print(
                "  - "
                f"camera={result.camera}, "
                f"bad_frames={result.bad_frames}, "
                f"bad_ratio={result.bad_ratio:.2%}, "
                f"strong_bad_frames={result.strong_bad_frames}, "
                f"checked_frames={result.checked_frames}, "
                f"max_score={result.max_score:.3f}, "
                f"video_path={rel_path}"
                f"{error_text}"
            )


def run_detection(args: argparse.Namespace) -> list[VideoResult]:
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    info = read_info(dataset_dir)
    tasks = build_video_tasks(
        dataset_dir=dataset_dir,
        info=info,
        camera=args.camera,
        episodes=args.episodes,
        max_episodes=args.max_episodes,
    )
    if not tasks:
        raise RuntimeError("没有生成任何待检测的视频任务")

    logger.info(
        "开始检测视频异常: dataset=%s, videos=%d, workers=%d, sample_stride=%d, "
        "threshold=%.3f, min_bad_frames=%d, min_bad_ratio=%.2f, strong_threshold=%.3f",
        dataset_dir,
        len(tasks),
        args.workers,
        args.sample_stride,
        args.threshold,
        args.min_bad_frames,
        args.min_bad_ratio,
        args.strong_threshold,
    )

    results: list[VideoResult] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(
                detect_video_glitches,
                task,
                args.sample_stride,
                args.threshold,
                args.strong_threshold,
            ): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            results.append(future.result())
            completed += 1
            if completed % 20 == 0 or completed == len(tasks):
                logger.info("已检测 %d/%d 个视频", completed, len(tasks))

    return sorted(results, key=lambda item: (item.episode_index, item.camera))


def main() -> int:
    args = parse_args()
    validate_args(args)
    dataset_dir = args.dataset_dir.expanduser().resolve()
    results = run_detection(args)
    print_summary(
        results=results,
        dataset_dir=dataset_dir,
        min_bad_frames=args.min_bad_frames,
        min_bad_ratio=args.min_bad_ratio,
        min_strong_bad_frames=args.min_strong_bad_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
