#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot 数据集 Episode 删除工具 (重构版)

功能说明：
    基于 FilteredDatasetCreator 实现，用于从 LeRobot 数据集中删除指定的 episode。
    该工具会自动处理 episode 重新索引、元数据更新和数据文件重组。

使用示例：
    # 删除 episode 1, 3, 5
    python tools/merge_multi_v21_lerobot_datasets/remove_episodes.py \
        --repo_id lerobot/pusht \
        --episodes 1,3,5 \
        --output_dir ./lerobot_pusht_cleaned

    # 使用文件指定要删除的 episode
    python tools/merge_multi_v21_lerobot_datasets/remove_episodes.py \
        --repo_id lerobot/pusht \
        --episodes delete_list.json \
        --output_dir ./lerobot_pusht_cleaned
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Set

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from dataset_creator.filtered_dataset_creator import FilteredDatasetCreator
except ImportError:
    # 尝试调整 sys.path 以确保能导入 dataset_creator
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from dataset_creator.filtered_dataset_creator import FilteredDatasetCreator
    except ImportError as e:
        print(f"错误: 无法导入必要的模块 ({e})")
        print("请确保已安装 lerobot 并在正确的目录下运行脚本")
        sys.exit(1)

def parse_episode_list(episodes_arg: str) -> Set[int]:
    """解析 episode 参数，返回要删除的 episode ID 集合"""
    episodes_to_delete = set()

    if not episodes_arg:
        return episodes_to_delete

    def parse_inline_list(value: str) -> Set[int]:
        parsed = set()
        parts = value.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                # 支持范围，如 10-20
                try:
                    start, end = map(int, part.split('-'))
                    parsed.update(range(start, end + 1))
                except ValueError:
                    logger.warning(f"无法解析范围: {part}")
            elif part.isdigit():
                parsed.add(int(part))
            else:
                logger.warning(f"忽略无效的 episode ID: {part}")
        return parsed

    # 明显是直接列表/范围时先解析，避免超长逗号列表被 Path.exists() 当文件名 stat。
    if ',' in episodes_arg or episodes_arg.replace('-', '').isdigit():
        return parse_inline_list(episodes_arg)

    # 尝试作为文件路径读取
    file_path = Path(episodes_arg)
    try:
        is_episode_file = file_path.exists() and file_path.is_file()
    except OSError as e:
        logger.warning(f"无法将参数作为文件路径检查 ({e})，尝试作为直接列表解析")
        is_episode_file = False

    if is_episode_file:
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix == '.json':
                    content = json.load(f)
                    if isinstance(content, list):
                        episodes_to_delete.update(content)
                    elif isinstance(content, dict):
                        # 尝试处理可能的字典格式，例如 {"episodes": [...]}
                        for v in content.values():
                            if isinstance(v, list):
                                episodes_to_delete.update(v)
                else:
                    # 假设是每行一个 ID 或逗号分隔
                    text = f.read()
                    parts = text.replace('\n', ',').split(',')
                    episodes_to_delete.update(int(p.strip()) for p in parts if p.strip().isdigit())
            logger.info(f"从文件加载了 {len(episodes_to_delete)} 个要删除的 episode")
            return episodes_to_delete
        except Exception as e:
            logger.warning(f"无法将参数作为文件读取 ({e})，尝试作为直接列表解析")

    # 尝试作为逗号分隔的字符串解析
    return parse_inline_list(episodes_arg)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
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


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def rows_by_contiguous_index(
    rows: list[dict[str, Any]],
    key: str,
    path: Path,
) -> dict[int, dict[str, Any]]:
    indexed = {}
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

    expected = list(range(len(rows)))
    if sorted(indexed) != expected:
        raise ValueError(f"{path} 的 {key} 必须从 0 连续编号")
    return indexed


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
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"数据集路径模板必须生成相对路径，当前为: {path}")
    return path


def integer_column_values(table: pa.Table, column_name: str, path: Path) -> np.ndarray:
    if column_name not in table.column_names:
        raise ValueError(f"{path} 缺少标准列: {column_name}")
    field = table.schema.field(column_name)
    if not pa.types.is_integer(field.type):
        raise ValueError(f"{path} 的 {column_name} 必须是整数列，当前为 {field.type}")
    column = table[column_name]
    if column.null_count:
        raise ValueError(f"{path} 的 {column_name} 包含空值")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)


def replace_integer_column(table: pa.Table, column_name: str, values: np.ndarray, path: Path) -> pa.Table:
    column_index = table.schema.get_field_index(column_name)
    if column_index < 0:
        raise ValueError(f"{path} 缺少标准列: {column_name}")
    field = table.schema.field(column_index)
    if not pa.types.is_integer(field.type):
        raise ValueError(f"{path} 的 {column_name} 必须是整数列，当前为 {field.type}")
    return table.set_column(column_index, field, pa.array(values, type=field.type))


def write_parquet_atomic(path: Path, table: pa.Table) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        pq.write_table(table, temp_path)
        shutil.copymode(path, temp_path)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def repair_output_parquet_indices(output_dir: Path) -> int:
    """将每个 Parquet 的 episode/frame/global index 修正为新数据集中的连续编号。"""
    output_dir = output_dir.resolve()
    info_path = output_dir / "meta" / "info.json"
    episodes_path = output_dir / "meta" / "episodes.jsonl"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    episodes = read_jsonl(episodes_path)
    episodes_by_index = rows_by_contiguous_index(episodes, "episode_index", episodes_path)

    chunks_size = info.get("chunks_size")
    if not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError(f"info.json chunks_size 必须是正整数，当前为 {chunks_size!r}")
    data_path_template = info.get("data_path")
    if not isinstance(data_path_template, str) or not data_path_template:
        raise ValueError("info.json 缺少有效的 data_path 模板")

    global_start_index = 0
    repaired_count = 0
    for position, (episode_index, episode) in enumerate(episodes_by_index.items(), start=1):
        length = episode.get("length")
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"episode {episode_index} 的 length 必须是正整数，当前为 {length!r}")
        parquet_path = output_dir / format_dataset_path(data_path_template, episode_index, chunks_size)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"输出 Parquet 不存在: {parquet_path}")

        index_table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "index"])
        if index_table.num_rows != length:
            raise ValueError(
                f"{parquet_path} 行数和 episodes.jsonl 不一致: parquet={index_table.num_rows}, metadata={length}"
            )
        expected_episode_indices = np.full(length, episode_index, dtype=np.int64)
        expected_frame_indices = np.arange(length, dtype=np.int64)
        expected_global_indices = np.arange(global_start_index, global_start_index + length, dtype=np.int64)
        current_episode_indices = integer_column_values(index_table, "episode_index", parquet_path)
        current_frame_indices = integer_column_values(index_table, "frame_index", parquet_path)
        current_global_indices = integer_column_values(index_table, "index", parquet_path)

        if not np.array_equal(current_frame_indices, expected_frame_indices):
            raise ValueError(f"{parquet_path} 的 frame_index 不是 0..{length - 1} 连续编号")
        if not (
            np.array_equal(current_episode_indices, expected_episode_indices)
            and np.array_equal(current_global_indices, expected_global_indices)
        ):
            table = pq.read_table(parquet_path)
            table = replace_integer_column(table, "episode_index", expected_episode_indices, parquet_path)
            table = replace_integer_column(table, "index", expected_global_indices, parquet_path)
            write_parquet_atomic(parquet_path, table)
            repaired_count += 1

        global_start_index += length
        if position % 100 == 0 or position == len(episodes_by_index):
            logger.info("已校验 Parquet index %d/%d", position, len(episodes_by_index))

    logger.info("Parquet index 修复完成: 修改 %d/%d 个文件", repaired_count, len(episodes_by_index))
    return repaired_count


def repair_output_info(output_dir: Path) -> dict[str, Any]:
    """根据输出文件和元数据重算 info.json，并在写入前校验数据集完整性。"""
    output_dir = output_dir.resolve()
    meta_dir = output_dir / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    required_paths = [info_path, episodes_path, stats_path, tasks_path]
    missing_meta = [path for path in required_paths if not path.is_file()]
    if missing_meta:
        raise FileNotFoundError("输出数据集缺少元数据文件: " + ", ".join(str(path) for path in missing_meta))

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    episodes = read_jsonl(episodes_path)
    episode_stats = read_jsonl(stats_path)
    tasks = read_jsonl(tasks_path)
    episodes_by_index = rows_by_contiguous_index(episodes, "episode_index", episodes_path)
    rows_by_contiguous_index(episode_stats, "episode_index", stats_path)
    rows_by_contiguous_index(tasks, "task_index", tasks_path)

    chunks_size = info.get("chunks_size")
    if not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError(f"info.json chunks_size 必须是正整数，当前为 {chunks_size!r}")
    data_path_template = info.get("data_path")
    if not isinstance(data_path_template, str) or not data_path_template:
        raise ValueError("info.json 缺少有效的 data_path 模板")

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.json features 必须是对象")
    video_keys = [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    video_path_template = info.get("video_path")
    if video_keys and (not isinstance(video_path_template, str) or not video_path_template):
        raise ValueError("数据集包含 video feature，但 info.json 缺少有效的 video_path 模板")

    total_frames = 0
    expected_files = []
    for episode_index, episode in episodes_by_index.items():
        length = episode.get("length")
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"episode {episode_index} 的 length 必须是正整数，当前为 {length!r}")
        total_frames += length
        expected_files.append(
            output_dir / format_dataset_path(data_path_template, episode_index, chunks_size)
        )
        for video_key in video_keys:
            expected_files.append(
                output_dir
                / format_dataset_path(
                    video_path_template,
                    episode_index,
                    chunks_size,
                    video_key,
                )
            )

    missing_files = [path for path in expected_files if not path.is_file()]
    if missing_files:
        preview = "\n".join(f"  - {path}" for path in missing_files[:20])
        suffix = "" if len(missing_files) <= 20 else f"\n  ... 另有 {len(missing_files) - 20} 个"
        raise FileNotFoundError(f"输出数据集缺少 Parquet/视频文件:\n{preview}{suffix}")

    total_episodes = len(episodes)
    repaired_info = dict(info)
    repaired_info["total_episodes"] = total_episodes
    repaired_info["total_frames"] = total_frames
    repaired_info["total_tasks"] = len(tasks)
    repaired_info["total_chunks"] = (total_episodes - 1) // chunks_size + 1 if total_episodes else 0
    repaired_info["total_videos"] = total_episodes * len(video_keys)
    repaired_info["splits"] = {"train": f"0:{total_episodes}"}

    changed_fields = {
        key: (info.get(key), repaired_info[key])
        for key in (
            "total_episodes",
            "total_frames",
            "total_tasks",
            "total_chunks",
            "total_videos",
            "splits",
        )
        if info.get(key) != repaired_info[key]
    }
    write_json_atomic(info_path, repaired_info)
    if changed_fields:
        logger.info("已修正输出 info.json: %s", changed_fields)
    else:
        logger.info("输出 info.json 派生字段已正确，无需修改")
    return repaired_info

def main():
    parser = argparse.ArgumentParser(
        description='LeRobot 数据集 Episode 删除工具 (Refactored)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--repo_id', type=str, required=True,
                       help='源数据集的 repo id (例如: lerobot/pusht)')
    parser.add_argument('--root', type=str, default=None,
                       help='源数据集根目录 (可选)')
    parser.add_argument('--episodes', type=str, required=True,
                       help='要删除的 episode 列表 (逗号分隔，支持范围 1-5，或 json/txt 文件路径)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='生成的清洗后数据集存放目录')
    parser.add_argument('--new_repo_id', type=str, default=None,
                       help='新数据集的 Repo ID (默认为 {repo_id}_filtered)')
    parser.add_argument('--temp_dir', type=str, default=None,
                       help='临时文件目录 (默认使用 output_dir 的父目录，避免占用 /tmp)')

    args = parser.parse_args()

    # 1. 加载源数据集
    logger.info(f"正在加载源数据集: {args.repo_id} ...")
    try:
        dataset = LeRobotDataset(args.repo_id, root=args.root)
        logger.info(f"源数据集加载成功，路径: {dataset.root}")
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return

    # 2. 确定要删除和保留的 episodes
    total_episodes = dataset.meta.total_episodes
    episodes_to_delete = parse_episode_list(args.episodes)

    # 过滤掉超出范围的 ID
    valid_delete = {eid for eid in episodes_to_delete if 0 <= eid < total_episodes}
    if len(valid_delete) < len(episodes_to_delete):
        logger.warning(f"忽略了 {len(episodes_to_delete) - len(valid_delete)} 个超出范围的 episode ID")

    episodes_to_delete = valid_delete

    if not episodes_to_delete:
        logger.warning("没有指定有效的要删除的 episode，程序退出")
        return

    # 计算要保留的 episodes
    episodes_to_keep = [i for i in range(total_episodes) if i not in episodes_to_delete]

    logger.info("统计:")
    logger.info(f"   - 原总 Episodes: {total_episodes}")
    logger.info(f"   - 将删除: {len(episodes_to_delete)}")
    logger.info(f"   - 将保留: {len(episodes_to_keep)}")

    if len(episodes_to_keep) == 0:
        logger.error("删除后没有剩余的 episode，操作终止")
        return

    # 3. 使用 FilteredDatasetCreator 创建新数据集
    output_dir = Path(args.output_dir)
    new_repo_id = args.new_repo_id or f"{args.repo_id}_filtered"

    logger.info(f"开始生成新数据集到 {output_dir} ...")

    creator = FilteredDatasetCreator(original_dataset=dataset)

    success = creator.create(
        new_repo_id=new_repo_id,
        selected_episodes=episodes_to_keep,
        push_to_hub=False,
        local_output_dir=output_dir,
        temp_dir=Path(args.temp_dir) if args.temp_dir else None,
    )

    if success:
        repair_output_parquet_indices(output_dir)
        repair_output_info(output_dir)
        logger.info(f"✅ 数据集处理完成! 输出目录: {output_dir}")
    else:
        logger.error("❌ 数据集创建失败")

if __name__ == "__main__":
    main()
