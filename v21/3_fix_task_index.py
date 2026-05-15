#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot 数据集 task_index 修复工具

功能说明：
    当数据集的 parquet 文件中存在 tasks.jsonl 中未定义的 task_index 时，
    会导致训练时出现 KeyError。本工具用于检查并修复此问题。

    支持两种修复模式：
    1. 统一模式（默认）：将所有 parquet 文件中的 task_index 统一为 0，
       适用于单任务数据集或不关心任务区分的场景。
    2. 补全模式：根据 parquet 中实际存在的 task_index，在 tasks.jsonl 中
       补全缺失的任务条目（使用默认占位任务描述或用户指定的描述）。

使用示例：
    # 检查数据集的 task_index 情况（不做修改）
    python tools/lerobot_dataset_tools/3_fix_task_index.py \
        --dataset_dir .cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long \
        --check_only

    # 统一所有 task_index 为 0（默认模式）
    python tools/lerobot_dataset_tools/3_fix_task_index.py \
        --dataset_dir .cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long

    # 补全 tasks.jsonl 中缺失的 task_index
    python tools/lerobot_dataset_tools/3_fix_task_index.py \
        --dataset_dir .cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long \
        --mode fill \
        --default_task "unknown task"
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_tasks_jsonl(tasks_path: Path) -> dict[int, str]:
    """从 tasks.jsonl 加载任务映射"""
    tasks = {}
    if not tasks_path.exists():
        logger.warning(f"tasks.jsonl 不存在: {tasks_path}")
        return tasks
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks[item["task_index"]] = item["task"]
    return tasks


def save_tasks_jsonl(tasks: dict[int, str], tasks_path: Path) -> None:
    """将任务映射写入 tasks.jsonl"""
    with open(tasks_path, "w", encoding="utf-8") as f:
        for task_index in sorted(tasks.keys()):
            item = {"task_index": task_index, "task": tasks[task_index]}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_episodes_jsonl(episodes_path: Path) -> list[dict]:
    """从 episodes.jsonl 加载 episode 信息"""
    episodes = []
    if not episodes_path.exists():
        logger.warning(f"episodes.jsonl 不存在: {episodes_path}")
        return episodes
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
    return episodes


def save_episodes_jsonl(episodes: list[dict], episodes_path: Path) -> None:
    """将 episode 信息写入 episodes.jsonl"""
    with open(episodes_path, "w", encoding="utf-8") as f:
        for ep in sorted(episodes, key=lambda x: x["episode_index"]):
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def find_parquet_files(dataset_dir: Path) -> list[Path]:
    """递归查找所有 parquet 文件"""
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return []
    return sorted(data_dir.rglob("*.parquet"))


def check_task_indices(dataset_dir: Path) -> tuple[dict[int, str], set[int], dict[str, set[int]]]:
    """
    检查数据集中的 task_index 情况。

    返回:
        tasks_in_meta: tasks.jsonl 中定义的 {task_index: task}
        all_task_indices: parquet 文件中所有出现的 task_index 集合
        file_task_map: 每个 parquet 文件中出现的 task_index {文件名: {task_indices}}
    """
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    tasks_in_meta = load_tasks_jsonl(tasks_path)

    parquet_files = find_parquet_files(dataset_dir)
    if not parquet_files:
        logger.error("未找到任何 parquet 文件")
        return tasks_in_meta, set(), {}

    all_task_indices = set()
    file_task_map = {}

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "task_index" not in df.columns:
            logger.warning(f"文件 {pf.name} 中没有 task_index 列，跳过")
            continue
        unique_indices = set(df["task_index"].unique().tolist())
        all_task_indices.update(unique_indices)
        file_task_map[str(pf.relative_to(dataset_dir))] = unique_indices

    return tasks_in_meta, all_task_indices, file_task_map


def print_check_report(
    tasks_in_meta: dict[int, str],
    all_task_indices: set[int],
    file_task_map: dict[str, set[int]],
) -> set[int]:
    """打印检查报告，返回缺失的 task_index 集合"""
    logger.info("=" * 60)
    logger.info("数据集 task_index 检查报告")
    logger.info("=" * 60)

    logger.info(f"\ntasks.jsonl 中定义的任务 ({len(tasks_in_meta)} 个):")
    for idx in sorted(tasks_in_meta.keys()):
        logger.info(f"  task_index={idx}: \"{tasks_in_meta[idx]}\"")

    logger.info(f"\nparquet 文件中出现的 task_index: {sorted(all_task_indices)}")

    missing = all_task_indices - set(tasks_in_meta.keys())
    extra = set(tasks_in_meta.keys()) - all_task_indices

    if missing:
        logger.warning(f"\n⚠️  缺失的 task_index (在 parquet 中存在但 tasks.jsonl 中未定义): {sorted(missing)}")
        for fname, indices in file_task_map.items():
            file_missing = indices & missing
            if file_missing:
                logger.warning(f"   {fname} 中包含缺失的 task_index: {sorted(file_missing)}")
    else:
        logger.info("\n✅ 所有 task_index 都已在 tasks.jsonl 中定义")

    if extra:
        logger.info(f"\nℹ️  多余的 task_index (在 tasks.jsonl 中定义但 parquet 中未使用): {sorted(extra)}")

    logger.info("=" * 60)
    return missing


def fix_unify(dataset_dir: Path, backup: bool = True) -> None:
    """
    统一模式修复：将所有 parquet 文件中的 task_index 统一为 0，
    并将 tasks.jsonl 和 episodes.jsonl 中的 task_index 也更新为 0。
    """
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"

    tasks_in_meta, all_task_indices, file_task_map = check_task_indices(dataset_dir)
    missing = print_check_report(tasks_in_meta, all_task_indices, file_task_map)

    if len(all_task_indices) <= 1 and 0 in all_task_indices and not missing:
        logger.info("所有 task_index 已经是 0，无需修复")
        return

    # 备份
    if backup:
        backup_dir = dataset_dir / "meta" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        if tasks_path.exists():
            shutil.copy2(tasks_path, backup_dir / "tasks.jsonl.bak")
            logger.info(f"已备份 tasks.jsonl -> {backup_dir / 'tasks.jsonl.bak'}")
        if episodes_path.exists():
            shutil.copy2(episodes_path, backup_dir / "episodes.jsonl.bak")
            logger.info(f"已备份 episodes.jsonl -> {backup_dir / 'episodes.jsonl.bak'}")

    # 1. 修复 parquet 文件
    parquet_files = find_parquet_files(dataset_dir)
    fixed_count = 0
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "task_index" not in df.columns:
            continue
        if df["task_index"].max() > 0 or df["task_index"].min() < 0:
            original_indices = sorted(df["task_index"].unique().tolist())
            if backup:
                shutil.copy2(pf, pf.with_suffix(".parquet.bak"))
            df["task_index"] = 0
            df.to_parquet(pf)
            fixed_count += 1
            logger.info(f"  已修复 {pf.name}: task_index {original_indices} -> [0]")

    logger.info(f"共修复了 {fixed_count} 个 parquet 文件")

    # 2. 修复 tasks.jsonl：只保留 task_index=0 的条目
    if tasks_in_meta:
        # 使用 task_index=0 的任务描述，如果没有则取第一个
        task_desc = tasks_in_meta.get(0, next(iter(tasks_in_meta.values())))
        save_tasks_jsonl({0: task_desc}, tasks_path)
        logger.info(f"  已更新 tasks.jsonl: 仅保留 task_index=0, task=\"{task_desc}\"")
    else:
        save_tasks_jsonl({0: "default task"}, tasks_path)
        logger.info("  已创建 tasks.jsonl: task_index=0, task=\"default task\"")

    # 3. 修复 episodes.jsonl 中的 task_index
    episodes = load_episodes_jsonl(episodes_path)
    if episodes:
        updated = False
        for ep in episodes:
            if "task_index" in ep and ep["task_index"] != 0:
                ep["task_index"] = 0
                updated = True
            # tasks 字段可能是列表
            if "tasks" in ep:
                ep["tasks"] = [0]
                updated = True
        if updated:
            save_episodes_jsonl(episodes, episodes_path)
            logger.info("  已更新 episodes.jsonl 中的 task_index 为 0")

    logger.info("\n✅ 统一模式修复完成！所有 task_index 已统一为 0")


def fix_fill(dataset_dir: Path, default_task: str = "unknown task", backup: bool = True) -> None:
    """
    补全模式修复：在 tasks.jsonl 中补全缺失的 task_index 条目。
    不修改 parquet 文件。
    """
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"

    tasks_in_meta, all_task_indices, file_task_map = check_task_indices(dataset_dir)
    missing = print_check_report(tasks_in_meta, all_task_indices, file_task_map)

    if not missing:
        logger.info("没有缺失的 task_index，无需修复")
        return

    # 备份
    if backup and tasks_path.exists():
        backup_dir = dataset_dir / "meta" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tasks_path, backup_dir / "tasks.jsonl.bak")
        logger.info(f"已备份 tasks.jsonl -> {backup_dir / 'tasks.jsonl.bak'}")

    # 补全缺失的 task_index
    for idx in sorted(missing):
        task_name = f"{default_task} {idx}"
        tasks_in_meta[idx] = task_name
        logger.info(f"  补全 task_index={idx}: \"{task_name}\"")

    save_tasks_jsonl(tasks_in_meta, tasks_path)
    logger.info(f"\n✅ 补全模式修复完成！共补全了 {len(missing)} 个 task_index")


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot 数据集 task_index 修复工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅检查
  python 3_fix_task_index.py --dataset_dir /path/to/dataset --check_only

  # 统一 task_index 为 0
  python 3_fix_task_index.py --dataset_dir /path/to/dataset

  # 补全 tasks.jsonl
  python 3_fix_task_index.py --dataset_dir /path/to/dataset --mode fill --default_task "fold towel"
        """,
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="数据集目录路径 (例如: .cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["unify", "fill"],
        default="unify",
        help="修复模式: unify=统一所有 task_index 为 0; fill=补全 tasks.jsonl 中缺失的条目 (默认: unify)",
    )
    parser.add_argument(
        "--default_task",
        type=str,
        default="unknown task",
        help="补全模式下，缺失 task_index 的默认任务描述 (默认: 'unknown task')",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="仅检查，不做任何修改",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="不备份原始文件（默认会备份）",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"数据集目录不存在: {dataset_dir}")
        return

    if not (dataset_dir / "meta").exists():
        logger.error(f"meta 目录不存在: {dataset_dir / 'meta'}")
        return

    if args.check_only:
        tasks_in_meta, all_task_indices, file_task_map = check_task_indices(dataset_dir)
        print_check_report(tasks_in_meta, all_task_indices, file_task_map)
        return

    backup = not args.no_backup

    if args.mode == "unify":
        fix_unify(dataset_dir, backup=backup)
    elif args.mode == "fill":
        fix_fill(dataset_dir, default_task=args.default_task, backup=backup)


if __name__ == "__main__":
    main()
