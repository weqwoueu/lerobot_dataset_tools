#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从本地 LeRobot v2.1 数据集中抽取闭区间 episode，生成独立的新数据集。

示例：
  python 26_extract_episode_range.py \
      --dataset-dir /path/to/source \
      --start-episode 10 \
      --end-episode 20

范围两端均包含。上例会抽取源 episode 10..20，并在输出数据集中重编号为 0..10。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from packaging.version import InvalidVersion, Version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CORE_META_FILES = {
    "info.json",
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "tasks.jsonl",
    "stats.json",
}
REQUIRED_PARQUET_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "index",
    "task_index",
)
STATS_FIELDS = ("min", "max", "mean", "std", "count")


@dataclass(frozen=True)
class VideoCopyPlan:
    video_key: str
    src_path: Path
    dst_relative_path: Path


@dataclass(frozen=True)
class EpisodeCopyPlan:
    src_episode_index: int
    dst_episode_index: int
    length: int
    global_start_index: int
    src_parquet_path: Path
    dst_parquet_relative_path: Path
    videos: tuple[VideoCopyPlan, ...]


@dataclass(frozen=True)
class ExtractionPlan:
    dataset_dir: Path
    output_dir: Path
    source_info: dict[str, Any]
    output_info: dict[str, Any]
    source_tasks: tuple[dict[str, Any], ...]
    output_tasks: tuple[dict[str, Any], ...]
    output_episodes: tuple[dict[str, Any], ...]
    output_episode_stats: tuple[dict[str, Any], ...]
    task_index_map: dict[int, int]
    episodes: tuple[EpisodeCopyPlan, ...]
    source_has_aggregate_stats: bool

    @property
    def total_frames(self) -> int:
        return sum(episode.length for episode in self.episodes)

    @property
    def total_videos(self) -> int:
        return sum(len(episode.videos) for episode in self.episodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抽取 LeRobot v2.1 数据集中的闭区间 episode，并生成连续重编号的新数据集。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        "--dataset_dir",
        dest="dataset_dir",
        type=Path,
        required=True,
        help="源 LeRobot v2.1 数据集目录。",
    )
    parser.add_argument(
        "--start-episode",
        "--start_episode",
        dest="start_episode",
        type=int,
        required=True,
        help="起始源 episode_index，包含该 episode。",
    )
    parser.add_argument(
        "--end-episode",
        "--end_episode",
        dest="end_episode",
        type=int,
        required=True,
        help="结束源 episode_index，包含该 episode。",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="输出目录。默认是源目录同级的 <name>_ep<start>_<end>。",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="完成预检并打印方案，但不创建输出。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="成功生成并校验新数据集后，替换已存在的输出目录。",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败: {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL 行必须是对象: {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def resolve_output_dir(
    dataset_dir: Path,
    output_dir: Path | None,
    start_episode: int,
    end_episode: int,
) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return dataset_dir.with_name(f"{dataset_dir.name}_ep{start_episode}_{end_episode}").resolve()


def is_path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_non_overlapping_paths(dataset_dir: Path, output_dir: Path) -> None:
    if dataset_dir == output_dir or is_path_within(output_dir, dataset_dir) or is_path_within(dataset_dir, output_dir):
        raise ValueError(f"源目录和输出目录不能相同或互相包含: source={dataset_dir}, output={output_dir}")


def validate_dataset_version(info: dict[str, Any]) -> None:
    raw_version = str(info.get("codebase_version", ""))
    try:
        version = Version(raw_version.removeprefix("v"))
    except InvalidVersion as exc:
        raise ValueError(f"info.json codebase_version 无效: {raw_version!r}") from exc
    if version.release[:2] != (2, 1):
        raise ValueError(f"仅支持 LeRobot v2.1 数据集，当前 codebase_version={raw_version!r}")


def validate_relative_template_path(path: Path, template_name: str) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{template_name} 必须生成数据集内的相对路径，当前为: {path}")
    return path


def format_dataset_path(
    template: str,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    values: dict[str, Any] = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    try:
        path = Path(template.format(**values))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"数据集路径模板无法格式化: {template!r}: {exc}") from exc
    return validate_relative_template_path(path, template)


def get_video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.json features 必须是对象")
    return [key for key, feature in features.items() if isinstance(feature, dict) and feature.get("dtype") == "video"]


def warn_if_source_info_integer_differs(info: dict[str, Any], field: str, expected_value: int) -> None:
    raw_value = info.get(field)
    try:
        actual_value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "源 info.json %s 不是有效整数，将按实际元数据继续: %r -> %d",
            field,
            raw_value,
            expected_value,
        )
        return
    if actual_value != expected_value:
        logger.warning(
            "源 info.json %s 与实际元数据不一致，将按实际元数据继续: %d -> %d",
            field,
            actual_value,
            expected_value,
        )


def rows_by_unique_integer_key(
    rows: list[dict[str, Any]],
    key: str,
    path: Path,
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if key not in row:
            raise ValueError(f"{path} 行缺少 {key}: {row}")
        try:
            index = int(row[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} 的 {key} 不是整数: {row[key]!r}") from exc
        if index in indexed:
            raise ValueError(f"{path} 存在重复 {key}: {index}")
        indexed[index] = row
    return indexed


def integer_column_values(table: pa.Table, column_name: str, path: Path) -> np.ndarray:
    field = table.schema.field(column_name)
    if not pa.types.is_integer(field.type):
        raise ValueError(f"{path} 的 {column_name} 必须是整数列，当前为 {field.type}")
    column = table[column_name]
    if column.null_count:
        raise ValueError(f"{path} 的 {column_name} 包含空值")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)


def validate_source_parquet(
    path: Path,
    episode_index: int,
    length: int,
    expected_global_start: int,
) -> set[int]:
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise ValueError(f"无法读取 Parquet: {path}: {exc}") from exc
    if parquet_file.metadata.num_rows != length:
        raise ValueError(
            f"{path} 行数和 episodes.jsonl 不一致: parquet={parquet_file.metadata.num_rows}, metadata={length}"
        )

    missing_columns = [name for name in REQUIRED_PARQUET_COLUMNS if name not in parquet_file.schema_arrow.names]
    if missing_columns:
        raise ValueError(f"{path} 缺少标准列: {missing_columns}")

    try:
        table = pq.read_table(path, columns=list(REQUIRED_PARQUET_COLUMNS))
    except Exception as exc:
        raise ValueError(f"无法读取 Parquet 标准列: {path}: {exc}") from exc

    expected_frame_indices = np.arange(length, dtype=np.int64)
    expected_global_indices = np.arange(expected_global_start, expected_global_start + length, dtype=np.int64)
    episode_indices = integer_column_values(table, "episode_index", path)
    frame_indices = integer_column_values(table, "frame_index", path)
    global_indices = integer_column_values(table, "index", path)
    task_indices = integer_column_values(table, "task_index", path)

    if not np.array_equal(episode_indices, np.full(length, episode_index, dtype=np.int64)):
        raise ValueError(f"{path} 的 episode_index 与源 episode {episode_index} 不一致")
    if not np.array_equal(frame_indices, expected_frame_indices):
        raise ValueError(f"{path} 的 frame_index 不是 0..{length - 1} 连续编号")
    if not np.array_equal(global_indices, expected_global_indices):
        raise ValueError(
            f"{path} 的 index 不符合源数据集连续全局编号，期望范围 "
            f"{expected_global_start}..{expected_global_start + length - 1}"
        )
    return {int(value) for value in task_indices.tolist()}


def validate_source_metadata(
    dataset_dir: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    validate_dataset_version(info)
    chunks_size = info.get("chunks_size")
    if not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError(f"info.json chunks_size 必须是正整数，当前为 {chunks_size!r}")
    if not isinstance(info.get("data_path"), str) or not info["data_path"]:
        raise ValueError("info.json 缺少有效的 data_path 模板")

    video_keys = get_video_keys(info)
    if video_keys and (not isinstance(info.get("video_path"), str) or not info["video_path"]):
        raise ValueError("数据集包含 video feature，但 info.json 缺少有效的 video_path 模板")

    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    episodes_by_index = rows_by_unique_integer_key(episodes, "episode_index", episodes_path)
    stats_by_index = rows_by_unique_integer_key(episode_stats, "episode_index", stats_path)
    tasks_by_index = rows_by_unique_integer_key(tasks, "task_index", tasks_path)

    expected_episode_indices = list(range(len(episodes)))
    if sorted(episodes_by_index) != expected_episode_indices:
        raise ValueError("源 episodes.jsonl 的 episode_index 必须从 0 连续编号")
    if sorted(stats_by_index) != expected_episode_indices:
        raise ValueError("源 episodes_stats.jsonl 必须为每个 episode 提供一条连续编号的统计")

    expected_task_indices = list(range(len(tasks)))
    if sorted(tasks_by_index) != expected_task_indices:
        raise ValueError("源 tasks.jsonl 的 task_index 必须从 0 连续编号")

    expected_total_chunks = math.ceil(len(episodes) / chunks_size)
    expected_total_videos = len(episodes) * len(video_keys)

    total_frames = 0
    task_names = set()
    for task_index, task_row in tasks_by_index.items():
        task_name = task_row.get("task")
        if not isinstance(task_name, str):
            raise ValueError(f"tasks.jsonl task_index={task_index} 缺少字符串 task")
        if task_name in task_names:
            raise ValueError(f"tasks.jsonl 存在重复 task 名称: {task_name!r}")
        task_names.add(task_name)

    for episode_index, episode_row in episodes_by_index.items():
        length = episode_row.get("length")
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"episode {episode_index} 的 length 必须是正整数，当前为 {length!r}")
        episode_tasks = episode_row.get("tasks")
        if not isinstance(episode_tasks, list) or not all(isinstance(item, str) for item in episode_tasks):
            raise ValueError(f"episode {episode_index} 的 tasks 必须是字符串列表")
        unknown_tasks = sorted(set(episode_tasks) - task_names)
        if unknown_tasks:
            raise ValueError(f"episode {episode_index} 引用了 tasks.jsonl 中不存在的任务: {unknown_tasks}")
        total_frames += length

    warn_if_source_info_integer_differs(info, "total_frames", total_frames)
    warn_if_source_info_integer_differs(info, "total_episodes", len(episodes))
    warn_if_source_info_integer_differs(info, "total_tasks", len(tasks))
    warn_if_source_info_integer_differs(info, "total_chunks", expected_total_chunks)
    warn_if_source_info_integer_differs(info, "total_videos", expected_total_videos)
    expected_splits = {"train": f"0:{len(episodes)}"}
    if info.get("splits") != expected_splits:
        logger.warning(
            "源 info.json splits 与标准全量 train split 不一致，将按实际 episode 范围继续: %r -> %r",
            info.get("splits"),
            expected_splits,
        )

    return episodes_by_index, stats_by_index, tasks_by_index


def build_extraction_plan(
    dataset_dir: Path,
    output_dir: Path,
    start_episode: int,
    end_episode: int,
) -> ExtractionPlan:
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    validate_non_overlapping_paths(dataset_dir, output_dir)

    required_paths = [
        dataset_dir / "meta" / "info.json",
        dataset_dir / "meta" / "episodes.jsonl",
        dataset_dir / "meta" / "episodes_stats.jsonl",
        dataset_dir / "meta" / "tasks.jsonl",
        dataset_dir / "data",
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("源数据集缺少必要文件/目录: " + ", ".join(str(path) for path in missing_paths))

    info = read_json(required_paths[0])
    episodes = read_jsonl(required_paths[1])
    episode_stats = read_jsonl(required_paths[2])
    tasks = read_jsonl(required_paths[3])
    episodes_by_index, stats_by_index, tasks_by_index = validate_source_metadata(
        dataset_dir,
        info,
        episodes,
        episode_stats,
        tasks,
    )

    total_episodes = len(episodes)
    if start_episode < 0 or end_episode < 0:
        raise ValueError("--start-episode 和 --end-episode 不能为负数")
    if start_episode > end_episode:
        raise ValueError("--start-episode 不能大于 --end-episode")
    if end_episode >= total_episodes:
        raise ValueError(
            f"episode 范围越界: [{start_episode}, {end_episode}]，源数据集有效范围为 [0, {total_episodes - 1}]"
        )

    chunks_size = int(info["chunks_size"])
    data_path_template = info["data_path"]
    video_keys = get_video_keys(info)
    video_path_template = info.get("video_path")
    source_global_starts: dict[int, int] = {}
    source_global_index = 0
    for episode_index in range(total_episodes):
        source_global_starts[episode_index] = source_global_index
        source_global_index += int(episodes_by_index[episode_index]["length"])

    episode_plans: list[EpisodeCopyPlan] = []
    used_task_indices: set[int] = set()
    source_parquet_paths: set[Path] = set()
    destination_parquet_paths: set[Path] = set()
    source_video_paths: set[Path] = set()
    destination_video_paths: set[Path] = set()
    output_global_index = 0
    for dst_episode_index, src_episode_index in enumerate(range(start_episode, end_episode + 1)):
        length = int(episodes_by_index[src_episode_index]["length"])
        src_data_relative = format_dataset_path(data_path_template, src_episode_index, chunks_size)
        dst_data_relative = format_dataset_path(data_path_template, dst_episode_index, chunks_size)
        if src_data_relative in source_parquet_paths:
            raise ValueError(f"data_path 模板让多个源 episode 指向同一文件: {src_data_relative}")
        if dst_data_relative in destination_parquet_paths:
            raise ValueError(f"data_path 模板让多个输出 episode 指向同一文件: {dst_data_relative}")
        source_parquet_paths.add(src_data_relative)
        destination_parquet_paths.add(dst_data_relative)
        src_parquet_path = dataset_dir / src_data_relative
        if not src_parquet_path.is_file():
            raise FileNotFoundError(f"源 Parquet 不存在: {src_parquet_path}")
        used_task_indices.update(
            validate_source_parquet(
                src_parquet_path,
                src_episode_index,
                length,
                source_global_starts[src_episode_index],
            )
        )

        videos: list[VideoCopyPlan] = []
        for video_key in video_keys:
            src_video_relative = format_dataset_path(
                video_path_template,
                src_episode_index,
                chunks_size,
                video_key,
            )
            dst_video_relative = format_dataset_path(
                video_path_template,
                dst_episode_index,
                chunks_size,
                video_key,
            )
            if src_video_relative in source_video_paths:
                raise ValueError(f"video_path 模板让多个源 episode/camera 指向同一文件: {src_video_relative}")
            if dst_video_relative in destination_video_paths:
                raise ValueError(f"video_path 模板让多个输出 episode/camera 指向同一文件: {dst_video_relative}")
            source_video_paths.add(src_video_relative)
            destination_video_paths.add(dst_video_relative)
            src_video_path = dataset_dir / src_video_relative
            if not src_video_path.is_file():
                raise FileNotFoundError(f"源视频不存在: {src_video_path}")
            videos.append(
                VideoCopyPlan(
                    video_key=video_key,
                    src_path=src_video_path,
                    dst_relative_path=dst_video_relative,
                )
            )

        episode_plans.append(
            EpisodeCopyPlan(
                src_episode_index=src_episode_index,
                dst_episode_index=dst_episode_index,
                length=length,
                global_start_index=output_global_index,
                src_parquet_path=src_parquet_path,
                dst_parquet_relative_path=dst_data_relative,
                videos=tuple(videos),
            )
        )
        output_global_index += length

    task_name_to_index = {str(row["task"]): index for index, row in tasks_by_index.items()}
    for src_episode_index in range(start_episode, end_episode + 1):
        for task_name in episodes_by_index[src_episode_index]["tasks"]:
            used_task_indices.add(task_name_to_index[task_name])

    unknown_task_indices = sorted(used_task_indices - set(tasks_by_index))
    if unknown_task_indices:
        raise ValueError(f"选中 Parquet 引用了 tasks.jsonl 中不存在的 task_index: {unknown_task_indices}")

    ordered_source_task_indices = [index for index in sorted(tasks_by_index) if index in used_task_indices]
    task_index_map = {old_index: new_index for new_index, old_index in enumerate(ordered_source_task_indices)}
    output_tasks = tuple(
        {**copy.deepcopy(tasks_by_index[old_index]), "task_index": task_index_map[old_index]}
        for old_index in ordered_source_task_indices
    )

    output_episodes = tuple(
        {
            **copy.deepcopy(episodes_by_index[episode.src_episode_index]),
            "episode_index": episode.dst_episode_index,
        }
        for episode in episode_plans
    )
    output_episode_stats = tuple(
        {
            **copy.deepcopy(stats_by_index[episode.src_episode_index]),
            "episode_index": episode.dst_episode_index,
        }
        for episode in episode_plans
    )

    output_info = copy.deepcopy(info)
    output_info["total_episodes"] = len(episode_plans)
    output_info["total_frames"] = output_global_index
    output_info["total_tasks"] = len(output_tasks)
    output_info["total_chunks"] = math.ceil(len(episode_plans) / chunks_size)
    output_info["total_videos"] = sum(len(episode.videos) for episode in episode_plans)
    output_info["splits"] = {"train": f"0:{len(episode_plans)}"}

    return ExtractionPlan(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        source_info=info,
        output_info=output_info,
        source_tasks=tuple(copy.deepcopy(tasks)),
        output_tasks=output_tasks,
        output_episodes=output_episodes,
        output_episode_stats=output_episode_stats,
        task_index_map=task_index_map,
        episodes=tuple(episode_plans),
        source_has_aggregate_stats=(dataset_dir / "meta" / "stats.json").is_file(),
    )


def replace_integer_column(table: pa.Table, column_name: str, values: np.ndarray, path: Path) -> pa.Table:
    column_index = table.schema.get_field_index(column_name)
    if column_index < 0:
        raise ValueError(f"{path} 缺少列: {column_name}")
    field = table.schema.field(column_index)
    if not pa.types.is_integer(field.type):
        raise ValueError(f"{path} 的 {column_name} 必须是整数列，当前为 {field.type}")
    array = pa.array(values, type=field.type)
    return table.set_column(column_index, field, array)


def write_episode_parquet(episode: EpisodeCopyPlan, staging_dir: Path, task_index_map: dict[int, int]) -> None:
    table = pq.read_table(episode.src_parquet_path)
    length = episode.length
    table = replace_integer_column(
        table,
        "episode_index",
        np.full(length, episode.dst_episode_index, dtype=np.int64),
        episode.src_parquet_path,
    )
    table = replace_integer_column(
        table,
        "index",
        np.arange(episode.global_start_index, episode.global_start_index + length, dtype=np.int64),
        episode.src_parquet_path,
    )
    old_task_indices = integer_column_values(table, "task_index", episode.src_parquet_path)
    try:
        new_task_indices = np.asarray([task_index_map[int(value)] for value in old_task_indices], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"{episode.src_parquet_path} 包含未规划的 task_index: {exc.args[0]}") from exc
    table = replace_integer_column(table, "task_index", new_task_indices, episode.src_parquet_path)

    output_path = staging_dir / episode.dst_parquet_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)


def is_backup_file(path: Path) -> bool:
    lowered_name = path.name.lower()
    return ".bak" in path.suffixes or "backup" in lowered_name


def copy_auxiliary_files(dataset_dir: Path, staging_dir: Path) -> None:
    for source_path in sorted(dataset_dir.iterdir()):
        if not source_path.is_file() or is_backup_file(source_path):
            continue
        if source_path.name == "norm_stats.json":
            logger.info("跳过可能与子集不一致的派生文件: %s", source_path)
            continue
        shutil.copy2(source_path, staging_dir / source_path.name)

    source_meta_dir = dataset_dir / "meta"
    destination_meta_dir = staging_dir / "meta"
    destination_meta_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_meta_dir.iterdir()):
        if not source_path.is_file() or source_path.name in CORE_META_FILES or is_backup_file(source_path):
            continue
        if source_path.suffix.lower() in {".json", ".jsonl"}:
            logger.info("跳过无法确认对子集仍有效的辅助元数据: %s", source_path)
            continue
        shutil.copy2(source_path, destination_meta_dir / source_path.name)


def aggregate_feature_stats(feature_stats: list[dict[str, Any]], feature_key: str) -> dict[str, Any]:
    for stats in feature_stats:
        missing_fields = [field for field in STATS_FIELDS if field not in stats]
        if missing_fields:
            raise ValueError(f"episode stats 的 {feature_key!r} 缺少字段: {missing_fields}")

    minima = [np.asarray(stats["min"]) for stats in feature_stats]
    maxima = [np.asarray(stats["max"]) for stats in feature_stats]
    means = np.stack([np.asarray(stats["mean"], dtype=np.float64) for stats in feature_stats])
    standard_deviations = np.stack([np.asarray(stats["std"], dtype=np.float64) for stats in feature_stats])
    counts = np.stack([np.asarray(stats["count"], dtype=np.float64) for stats in feature_stats])
    total_count = counts.sum(axis=0)
    if np.any(total_count <= 0):
        raise ValueError(f"episode stats 的 {feature_key!r} count 总和必须大于 0")

    broadcast_counts = counts
    while broadcast_counts.ndim < means.ndim:
        broadcast_counts = np.expand_dims(broadcast_counts, axis=-1)
    total_mean = (means * broadcast_counts).sum(axis=0) / total_count
    delta_means = means - total_mean
    total_variance = (
        ((standard_deviations**2 + delta_means**2) * broadcast_counts).sum(axis=0) / total_count
    )

    return {
        "min": np.min(np.stack(minima), axis=0).tolist(),
        "max": np.max(np.stack(maxima), axis=0).tolist(),
        "mean": total_mean.tolist(),
        "std": np.sqrt(total_variance).tolist(),
        "count": total_count.astype(np.int64).tolist(),
    }


def aggregate_episode_stats(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    payloads = [row.get("stats") for row in rows]
    if not all(isinstance(payload, dict) for payload in payloads):
        raise ValueError("episodes_stats.jsonl 每行都必须包含 stats 对象")
    feature_keys = sorted({key for payload in payloads for key in payload})
    return {
        feature_key: aggregate_feature_stats(
            [payload[feature_key] for payload in payloads if feature_key in payload],
            feature_key,
        )
        for feature_key in feature_keys
    }


def write_staging_dataset(plan: ExtractionPlan, staging_dir: Path) -> None:
    copy_auxiliary_files(plan.dataset_dir, staging_dir)
    for position, episode in enumerate(plan.episodes, start=1):
        write_episode_parquet(episode, staging_dir, plan.task_index_map)
        for video in episode.videos:
            destination = staging_dir / video.dst_relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video.src_path, destination)
        if position % 10 == 0 or position == len(plan.episodes):
            logger.info("已写入 episode %d/%d", position, len(plan.episodes))

    write_json(staging_dir / "meta" / "info.json", plan.output_info)
    write_jsonl(staging_dir / "meta" / "episodes.jsonl", plan.output_episodes)
    write_jsonl(staging_dir / "meta" / "episodes_stats.jsonl", plan.output_episode_stats)
    write_jsonl(staging_dir / "meta" / "tasks.jsonl", plan.output_tasks)
    if plan.source_has_aggregate_stats:
        write_json(staging_dir / "meta" / "stats.json", aggregate_episode_stats(plan.output_episode_stats))


def validate_output_dataset(plan: ExtractionPlan, staging_dir: Path) -> None:
    info = read_json(staging_dir / "meta" / "info.json")
    episodes = read_jsonl(staging_dir / "meta" / "episodes.jsonl")
    episode_stats = read_jsonl(staging_dir / "meta" / "episodes_stats.jsonl")
    tasks = read_jsonl(staging_dir / "meta" / "tasks.jsonl")

    expected_episode_indices = list(range(len(plan.episodes)))
    if [int(row["episode_index"]) for row in episodes] != expected_episode_indices:
        raise ValueError("输出 episodes.jsonl 的 episode_index 不连续")
    if [int(row["episode_index"]) for row in episode_stats] != expected_episode_indices:
        raise ValueError("输出 episodes_stats.jsonl 的 episode_index 不连续")
    if [int(row["task_index"]) for row in tasks] != list(range(len(tasks))):
        raise ValueError("输出 tasks.jsonl 的 task_index 不连续")

    expected_info_values = {
        "total_episodes": len(plan.episodes),
        "total_frames": plan.total_frames,
        "total_tasks": len(plan.output_tasks),
        "total_chunks": math.ceil(len(plan.episodes) / int(info["chunks_size"])),
        "total_videos": plan.total_videos,
    }
    for key, expected_value in expected_info_values.items():
        if int(info.get(key, -1)) != expected_value:
            raise ValueError(f"输出 info.json {key} 不正确: {info.get(key)!r} != {expected_value}")
    expected_splits = {"train": f"0:{len(plan.episodes)}"}
    if info.get("splits") != expected_splits:
        raise ValueError(f"输出 info.json splits 不正确: {info.get('splits')!r} != {expected_splits!r}")

    for episode in plan.episodes:
        destination = staging_dir / episode.dst_parquet_relative_path
        if not destination.is_file():
            raise FileNotFoundError(f"输出 Parquet 不存在: {destination}")
        output_table = pq.read_table(destination, columns=list(REQUIRED_PARQUET_COLUMNS))
        source_table = pq.read_table(episode.src_parquet_path, columns=["frame_index", "timestamp"])
        if output_table.num_rows != episode.length:
            raise ValueError(f"输出 Parquet 行数错误: {destination}")

        expected_episode_values = np.full(episode.length, episode.dst_episode_index, dtype=np.int64)
        expected_global_values = np.arange(
            episode.global_start_index,
            episode.global_start_index + episode.length,
            dtype=np.int64,
        )
        if not np.array_equal(integer_column_values(output_table, "episode_index", destination), expected_episode_values):
            raise ValueError(f"输出 Parquet episode_index 错误: {destination}")
        if not np.array_equal(integer_column_values(output_table, "index", destination), expected_global_values):
            raise ValueError(f"输出 Parquet index 错误: {destination}")
        if not output_table["frame_index"].equals(source_table["frame_index"]):
            raise ValueError(f"输出 Parquet frame_index 被意外修改: {destination}")
        if not output_table["timestamp"].equals(source_table["timestamp"]):
            raise ValueError(f"输出 Parquet timestamp 被意外修改: {destination}")
        output_task_indices = set(integer_column_values(output_table, "task_index", destination).tolist())
        if not output_task_indices.issubset(set(range(len(tasks)))):
            raise ValueError(f"输出 Parquet task_index 超出 tasks.jsonl 范围: {destination}")

        for video in episode.videos:
            output_video = staging_dir / video.dst_relative_path
            if not output_video.is_file():
                raise FileNotFoundError(f"输出视频不存在: {output_video}")
            if output_video.stat().st_size != video.src_path.stat().st_size:
                raise ValueError(f"输出视频大小和源文件不一致: {output_video}")

    if plan.source_has_aggregate_stats and not (staging_dir / "meta" / "stats.json").is_file():
        raise FileNotFoundError("源数据集包含 meta/stats.json，但输出未生成聚合统计")
    if (staging_dir / "norm_stats.json").exists():
        raise ValueError("输出不应复制可能失真的 norm_stats.json")


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def publish_staging_directory(staging_dir: Path, output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        os.replace(staging_dir, output_dir)
        return
    if not force:
        raise FileExistsError(f"输出目录已存在，请换一个路径或加 --force: {output_dir}")

    backup_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-", dir=output_dir.parent))
    backup_dir.rmdir()
    os.replace(output_dir, backup_dir)
    try:
        os.replace(staging_dir, output_dir)
    except Exception:
        os.replace(backup_dir, output_dir)
        raise
    try:
        remove_path(backup_dir)
    except OSError as exc:
        logger.warning("新数据集已发布，但旧输出备份清理失败: %s: %s", backup_dir, exc)


def log_plan(plan: ExtractionPlan) -> None:
    first_episode = plan.episodes[0]
    last_episode = plan.episodes[-1]
    logger.info("抽取方案:")
    logger.info("  源数据集: %s", plan.dataset_dir)
    logger.info("  输出目录: %s", plan.output_dir)
    logger.info(
        "  episode 闭区间: %d..%d -> 0..%d（共 %d 个）",
        first_episode.src_episode_index,
        last_episode.src_episode_index,
        last_episode.dst_episode_index,
        len(plan.episodes),
    )
    logger.info("  总帧数: %d", plan.total_frames)
    logger.info("  任务数: %d -> %d", len(plan.source_tasks), len(plan.output_tasks))
    logger.info("  视频数: %d（逐文件复制，不重新编码）", plan.total_videos)
    logger.info(
        "  输出 info.json 派生字段: total_episodes=%d, total_frames=%d, total_tasks=%d, "
        "total_chunks=%d, total_videos=%d, splits=%s",
        plan.output_info["total_episodes"],
        plan.output_info["total_frames"],
        plan.output_info["total_tasks"],
        plan.output_info["total_chunks"],
        plan.output_info["total_videos"],
        plan.output_info["splits"],
    )
    logger.info("  全局 index: 0..%d", plan.total_frames - 1)
    logger.info("  meta/stats.json: %s", "重新聚合" if plan.source_has_aggregate_stats else "源数据集不存在，不生成")
    logger.info("  norm_stats.json: 不复制")


def process_dataset(
    dataset_dir: Path,
    start_episode: int,
    end_episode: int,
    output_dir: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> Path:
    dataset_dir = dataset_dir.resolve()
    resolved_output_dir = resolve_output_dir(dataset_dir, output_dir, start_episode, end_episode)
    if resolved_output_dir.exists() and not force and not dry_run:
        raise FileExistsError(f"输出目录已存在，请换一个路径或加 --force: {resolved_output_dir}")

    plan = build_extraction_plan(
        dataset_dir=dataset_dir,
        output_dir=resolved_output_dir,
        start_episode=start_episode,
        end_episode=end_episode,
    )
    log_plan(plan)
    if dry_run:
        logger.info("dry-run 完成：未创建输出目录。")
        return resolved_output_dir

    resolved_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{resolved_output_dir.name}.extract-",
            dir=resolved_output_dir.parent,
        )
    )
    try:
        write_staging_dataset(plan, staging_dir)
        validate_output_dataset(plan, staging_dir)
        publish_staging_directory(staging_dir, resolved_output_dir, force)
    except Exception:
        remove_path(staging_dir)
        raise

    logger.info("完成: %s", resolved_output_dir)
    return resolved_output_dir


def main() -> int:
    args = parse_args()
    process_dataset(
        dataset_dir=args.dataset_dir,
        start_episode=args.start_episode,
        end_episode=args.end_episode,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
