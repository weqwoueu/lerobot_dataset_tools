#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 piper 叠衣服数据集对齐到 Task_A/dagger 的 LeRobot v2.1 元数据格式。

默认不会修改源数据集，而是生成一个新的输出目录。脚本只做 key/prompt/meta
对齐和文件复制，不转码视频，也不改写 parquet 内容。

示例:
  .venv/bin/python tools/lerobot_dataset_tools/12_convert_piper_to_task_a_dagger.py --dry_run

  .venv/bin/python tools/lerobot_dataset_tools/12_convert_piper_to_task_a_dagger.py \
      --output_dir data/standard_Task_A/dagger/piper_fold_tshirt_green_small_nonidle_task_a_aligned
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_SOURCE_DIR = Path("data/standard_Task_A/dagger/piper_fold_tshirt_green_small_nonidle")
DEFAULT_TARGET_REFERENCE_DIR = Path("data/Task_A/dagger")
DEFAULT_OUTPUT_SUFFIX = "_task_a_aligned"
TASK_PROMPT = "fold the cloth, Advantage: positive"

CAMERA_KEY_MAP = {
    "observation.images.cam_high": "observation.images.top_head",
    "observation.images.cam_left_wrist": "observation.images.hand_left",
    "observation.images.cam_right_wrist": "observation.images.hand_right",
}

META_FILES_TO_COPY = {
    "nonidle_filter_stats.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 piper_fold_tshirt_green_small_nonidle 转换为可与 Task_A/dagger 合并的格式。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"源 LeRobot 数据集目录，默认 {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--target_reference_dir",
        type=Path,
        default=DEFAULT_TARGET_REFERENCE_DIR,
        help=f"用于对齐字段的目标参考数据集目录，默认 {DEFAULT_TARGET_REFERENCE_DIR}",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="输出数据集目录。默认是源目录同级的 <source_name>_task_a_aligned。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录已存在，先删除后重新生成。",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印转换计划，不写入任何文件。",
    )
    return parser.parse_args()


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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_output_dir(source_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return source_dir.with_name(f"{source_dir.name}{DEFAULT_OUTPUT_SUFFIX}")


def require_dataset_dir(dataset_dir: Path, label: str) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"{label}不存在: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"{label}不是目录: {dataset_dir}")
    for rel_path in ("meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl", "data", "videos"):
        path = dataset_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"{label}缺少必要路径: {path}")


def video_feature_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    return [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def convert_info(source_info: dict[str, Any], target_info: dict[str, Any]) -> dict[str, Any]:
    converted = dict(source_info)
    converted["robot_type"] = target_info.get("robot_type", converted.get("robot_type"))

    source_features = source_info.get("features", {})
    target_features = target_info.get("features", {})
    converted_features: dict[str, Any] = {}

    target_video_features = {
        key: target_features[key]
        for key in target_features
        if isinstance(target_features[key], dict) and target_features[key].get("dtype") == "video"
    }

    for key, feature in source_features.items():
        new_key = CAMERA_KEY_MAP.get(key, key)
        new_feature = json.loads(json.dumps(feature))

        if key in CAMERA_KEY_MAP:
            target_feature = target_video_features.get(new_key)
            if target_feature is not None:
                source_codec = new_feature.get("info", {}).get("video.codec")
                new_feature["names"] = target_feature.get("names")
                new_feature["shape"] = target_feature.get("shape", new_feature.get("shape"))
                new_feature["info"] = json.loads(json.dumps(target_feature.get("info", new_feature.get("info", {}))))
                if source_codec is not None:
                    new_feature["info"]["video.codec"] = source_codec

        if new_key in {"observation.state", "action"}:
            new_feature["names"] = target_features.get(new_key, {}).get("names")

        converted_features[new_key] = new_feature

    converted["features"] = converted_features
    return converted


def convert_episodes(source_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(source_dir / "meta" / "episodes.jsonl")
    converted = []
    for row in rows:
        new_row = dict(row)
        new_row["tasks"] = [TASK_PROMPT]
        new_row["task_index"] = 0
        converted.append(new_row)
    return converted


def convert_episode_stats(source_dir: Path) -> list[dict[str, Any]]:
    stats_path = source_dir / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        return []

    rows = read_jsonl(stats_path)
    converted = []
    for row in rows:
        new_row = dict(row)
        stats = new_row.get("stats")
        if isinstance(stats, dict):
            new_stats = {}
            for key, value in stats.items():
                new_stats[CAMERA_KEY_MAP.get(key, key)] = value
            new_row["stats"] = new_stats
        converted.append(new_row)
    return converted


def copy_data_dir(source_dir: Path, output_dir: Path) -> None:
    shutil.copytree(source_dir / "data", output_dir / "data")


def copy_videos_with_renamed_keys(source_dir: Path, output_dir: Path) -> None:
    source_videos_dir = source_dir / "videos"
    output_videos_dir = output_dir / "videos"
    output_videos_dir.mkdir(parents=True, exist_ok=True)

    for chunk_dir in sorted(source_videos_dir.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        dst_chunk_dir = output_videos_dir / chunk_dir.name
        dst_chunk_dir.mkdir(parents=True, exist_ok=True)

        for item in sorted(chunk_dir.iterdir()):
            if not item.is_dir():
                continue
            dst_key = CAMERA_KEY_MAP.get(item.name, item.name)
            shutil.copytree(item, dst_chunk_dir / dst_key)


def write_converted_meta(source_dir: Path, target_reference_dir: Path, output_dir: Path) -> None:
    source_info = read_json(source_dir / "meta" / "info.json")
    target_info = read_json(target_reference_dir / "meta" / "info.json")
    converted_info = convert_info(source_info, target_info)

    write_json(output_dir / "meta" / "info.json", converted_info)
    write_jsonl(output_dir / "meta" / "tasks.jsonl", [{"task_index": 0, "task": TASK_PROMPT}])
    write_jsonl(output_dir / "meta" / "episodes.jsonl", convert_episodes(source_dir))
    write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", convert_episode_stats(source_dir))

    for file_name in META_FILES_TO_COPY:
        source_path = source_dir / "meta" / file_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / "meta" / file_name)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，请换目录或使用 --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def count_parquet_files(dataset_dir: Path) -> int:
    return sum(1 for _ in (dataset_dir / "data").rglob("*.parquet"))


def count_video_files(dataset_dir: Path) -> int:
    return sum(1 for _ in (dataset_dir / "videos").rglob("*.mp4"))


def validate_output(output_dir: Path) -> None:
    required_files = [
        output_dir / "meta" / "info.json",
        output_dir / "meta" / "episodes.jsonl",
        output_dir / "meta" / "tasks.jsonl",
        output_dir / "meta" / "episodes_stats.jsonl",
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"输出缺少必要文件: {path}")

    info = read_json(output_dir / "meta" / "info.json")
    total_episodes = int(info["total_episodes"])
    video_keys = video_feature_keys(info)

    parquet_count = count_parquet_files(output_dir)
    if parquet_count != total_episodes:
        raise RuntimeError(f"parquet 数量不匹配: actual={parquet_count}, expected={total_episodes}")

    video_count = count_video_files(output_dir)
    expected_video_count = total_episodes * len(video_keys)
    if video_count != expected_video_count:
        raise RuntimeError(f"视频数量不匹配: actual={video_count}, expected={expected_video_count}")

    for video_key in video_keys:
        key_dirs = sorted((output_dir / "videos").glob(f"chunk-*/{video_key}"))
        if not key_dirs:
            raise RuntimeError(f"视频 key 没有对应目录: {video_key}")

    task_rows = read_jsonl(output_dir / "meta" / "tasks.jsonl")
    if task_rows != [{"task_index": 0, "task": TASK_PROMPT}]:
        raise RuntimeError(f"tasks.jsonl 内容不符合预期: {task_rows}")

    episode_rows = read_jsonl(output_dir / "meta" / "episodes.jsonl")
    if len(episode_rows) != total_episodes:
        raise RuntimeError(f"episodes.jsonl 行数不匹配: actual={len(episode_rows)}, expected={total_episodes}")
    bad_episodes = [
        row.get("episode_index")
        for row in episode_rows
        if row.get("task_index") != 0 or row.get("tasks") != [TASK_PROMPT]
    ]
    if bad_episodes:
        raise RuntimeError(f"发现未对齐任务字段的 episode: {bad_episodes[:20]}")


def log_plan(source_dir: Path, target_reference_dir: Path, output_dir: Path) -> None:
    source_info = read_json(source_dir / "meta" / "info.json")
    target_info = read_json(target_reference_dir / "meta" / "info.json")

    logger.info("源数据集: %s", source_dir)
    logger.info("目标参考数据集: %s", target_reference_dir)
    logger.info("输出目录: %s", output_dir)
    logger.info("robot_type: %s -> %s", source_info.get("robot_type"), target_info.get("robot_type"))
    logger.info("prompt: %s", TASK_PROMPT)
    logger.info("相机 key 映射:")
    for src_key, dst_key in CAMERA_KEY_MAP.items():
        logger.info("  %s -> %s", src_key, dst_key)
    logger.info("源视频 codec 将保留在 info.json 中，不执行视频转码")


def convert_dataset(args: argparse.Namespace) -> None:
    source_dir = args.source_dir.resolve()
    target_reference_dir = args.target_reference_dir.resolve()
    output_dir = resolve_output_dir(source_dir, args.output_dir).resolve()

    require_dataset_dir(source_dir, "源数据集")
    require_dataset_dir(target_reference_dir, "目标参考数据集")
    log_plan(source_dir, target_reference_dir, output_dir)

    if args.dry_run:
        logger.info("dry_run 已启用，不写入文件")
        return

    prepare_output_dir(output_dir, args.overwrite)
    copy_data_dir(source_dir, output_dir)
    copy_videos_with_renamed_keys(source_dir, output_dir)
    write_converted_meta(source_dir, target_reference_dir, output_dir)
    validate_output(output_dir)
    logger.info("转换完成并通过校验: %s", output_dir)


def main() -> int:
    args = parse_args()
    convert_dataset(args)
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
