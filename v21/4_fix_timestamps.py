#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot 数据集 timestamps 修复工具

功能说明：
    当数据集的 parquet 文件中 episode 内部时间戳不连续（如合并数据集后时间戳未对齐），
    会导致 lerobot 加载时报错：
        ValueError: One or several timestamps unexpectedly violate the tolerance inside episode range.
    本工具会为每个 episode 重新生成严格递增的时间戳：0, 1/fps, 2/fps, ...

使用示例：
    # 检查数据集的时间戳情况（不做修改）
    python tools/lerobot_dataset_tools/4_fix_timestamps.py \
        --dataset_dir .cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long \
        --check_only

    # 修复时间戳
    python tools/lerobot_dataset_tools/4_fix_timestamps.py \
        --dataset_dir .cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long

    # 仅修复指定 episode
    python tools/lerobot_dataset_tools/4_fix_timestamps.py \
        --dataset_dir .cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long \
        --episodes 21-67
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_info(dataset_dir: Path) -> dict:
    """加载 info.json"""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json 不存在: {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_parquet_files(dataset_dir: Path) -> list[Path]:
    """递归查找所有 parquet 文件，按名称排序"""
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return []
    return sorted(data_dir.rglob("*.parquet"))


def parse_episode_list(episodes_arg: str | None, total_episodes: int) -> set[int] | None:
    """解析 episode 参数，返回要修复的 episode 集合，None 表示全部"""
    if episodes_arg is None:
        return None

    episodes = set()
    parts = episodes_arg.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                episodes.update(range(start, end + 1))
            except ValueError:
                logger.warning(f"无法解析范围: {part}")
        elif part.isdigit():
            episodes.add(int(part))
        else:
            logger.warning(f"忽略无效的 episode ID: {part}")

    # 过滤掉超出范围的
    valid = {e for e in episodes if 0 <= e < total_episodes}
    if len(valid) < len(episodes):
        logger.warning(f"忽略了 {len(episodes) - len(valid)} 个超出范围的 episode ID")
    return valid


def check_timestamps(
    dataset_dir: Path,
    fps: int,
    tolerance_s: float = 1e-4,
    target_episodes: set[int] | None = None,
) -> dict[int, list[dict]]:
    """
    检查各 episode 的时间戳是否连续。

    返回: {episode_index: [问题详情列表]}
    """
    parquet_files = find_parquet_files(dataset_dir)
    if not parquet_files:
        return {}

    problems = {}
    expected_diff = 1.0 / fps

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "timestamp" not in df.columns or "episode_index" not in df.columns:
            logger.warning(f"文件 {pf.name} 缺少 timestamp 或 episode_index 列，跳过")
            continue

        for ep_idx, group in df.groupby("episode_index"):
            if target_episodes is not None and ep_idx not in target_episodes:
                continue

            timestamps = group["timestamp"].values
            if len(timestamps) < 2:
                continue

            diffs = np.diff(timestamps)
            bad_indices = np.where(np.abs(diffs - expected_diff) > tolerance_s)[0]

            if len(bad_indices) > 0:
                ep_problems = []
                for bi in bad_indices[:5]:  # 每个 episode 最多报告 5 个问题
                    ep_problems.append({
                        "frame_idx": int(bi),
                        "timestamps": [float(timestamps[bi]), float(timestamps[bi + 1])],
                        "actual_diff": float(diffs[bi]),
                        "expected_diff": expected_diff,
                    })
                problems[ep_idx] = ep_problems

    return problems


def print_check_report(problems: dict[int, list[dict]], fps: int) -> None:
    """打印检查报告"""
    logger.info("=" * 70)
    logger.info("数据集 timestamps 检查报告")
    logger.info(f"FPS: {fps}, 期望帧间隔: {1.0 / fps:.6f}s, 容差: 1e-4s")
    logger.info("=" * 70)

    if not problems:
        logger.info("\n✅ 所有 episode 的时间戳均正常")
        return

    logger.warning(f"\n⚠️  共 {len(problems)} 个 episode 存在时间戳问题:")
    for ep_idx in sorted(problems.keys()):
        ep_issues = problems[ep_idx]
        logger.warning(f"\n  Episode {ep_idx}:")
        for issue in ep_issues:
            ts = issue["timestamps"]
            logger.warning(
                f"    帧 {issue['frame_idx']}: timestamp [{ts[0]:.5f}] -> [{ts[1]:.5f}], "
                f"diff={issue['actual_diff']:.5f} (期望 {issue['expected_diff']:.6f})"
            )
        if len(problems[ep_idx]) == 5:
            logger.warning(f"    ... (仅显示前 5 个问题)")

    logger.info("\n" + "=" * 70)
    problem_episodes = sorted(problems.keys())
    if len(problem_episodes) <= 20:
        logger.info(f"问题 episode 列表: {problem_episodes}")
    else:
        logger.info(f"问题 episode 范围: {problem_episodes[0]} - {problem_episodes[-1]} (共 {len(problem_episodes)} 个)")
    logger.info("=" * 70)


def fix_timestamps(
    dataset_dir: Path,
    fps: int,
    target_episodes: set[int] | None = None,
    backup: bool = True,
) -> int:
    """
    修复时间戳：为每个 episode 重新生成严格递增的时间戳 0, 1/fps, 2/fps, ...

    返回修复的文件数量。
    """
    parquet_files = find_parquet_files(dataset_dir)
    if not parquet_files:
        return 0

    expected_diff = 1.0 / fps
    fixed_file_count = 0

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "timestamp" not in df.columns or "episode_index" not in df.columns:
            continue

        modified = False

        for ep_idx, group in df.groupby("episode_index"):
            if target_episodes is not None and ep_idx not in target_episodes:
                continue

            timestamps = group["timestamp"].values
            if len(timestamps) < 2:
                continue

            # 检查是否需要修复
            diffs = np.diff(timestamps)
            if np.all(np.abs(diffs - expected_diff) <= 1e-4):
                continue

            # 生成新的时间戳: 0, 1/fps, 2/fps, ...
            new_timestamps = np.arange(len(timestamps), dtype=np.float32) * expected_diff
            # 使用 round 避免浮点精度问题，保留 5 位小数
            new_timestamps = np.round(new_timestamps, decimals=5)

            df.loc[group.index, "timestamp"] = new_timestamps
            modified = True
            logger.info(
                f"  修复 {pf.name} episode {ep_idx}: {len(timestamps)} 帧, "
                f"原始时间戳范围 [{timestamps[0]:.3f}, {timestamps[-1]:.3f}] -> "
                f"[{new_timestamps[0]:.3f}, {new_timestamps[-1]:.3f}]"
            )

        if modified:
            if backup:
                backup_path = pf.with_suffix(".parquet.ts_bak")
                if not backup_path.exists():
                    shutil.copy2(pf, backup_path)
                    logger.info(f"  已备份 {pf.name} -> {backup_path.name}")

            df.to_parquet(pf)
            fixed_file_count += 1

    return fixed_file_count


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot 数据集 timestamps 修复工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅检查
  python 4_fix_timestamps.py --dataset_dir /path/to/dataset --check_only

  # 修复所有 episode
  python 4_fix_timestamps.py --dataset_dir /path/to/dataset

  # 仅修复指定 episode
  python 4_fix_timestamps.py --dataset_dir /path/to/dataset --episodes 21-67
        """,
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="数据集目录路径 (例如: .cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long)",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="仅检查，不做任何修改",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="仅处理指定的 episode (逗号分隔，支持范围如 21-67)。不指定则处理全部",
    )
    parser.add_argument(
        "--tolerance_s",
        type=float,
        default=1e-4,
        help="时间戳容差 (秒), 默认 1e-4",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="不备份原始 parquet 文件（默认会备份为 .parquet.ts_bak）",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"数据集目录不存在: {dataset_dir}")
        return

    # 读取 fps
    info = load_info(dataset_dir)
    fps = info.get("fps")
    if fps is None:
        logger.error("info.json 中未找到 fps 字段")
        return
    logger.info(f"数据集 FPS: {fps}")

    # 解析目标 episode
    total_episodes = info.get("total_episodes", 999999)
    target_episodes = parse_episode_list(args.episodes, total_episodes)
    if target_episodes is not None:
        logger.info(f"目标 episode: {sorted(target_episodes)}")
    else:
        logger.info("目标: 所有 episode")

    # 检查
    logger.info("\n正在检查时间戳...")
    problems = check_timestamps(dataset_dir, fps, args.tolerance_s, target_episodes)
    print_check_report(problems, fps)

    if args.check_only:
        return

    if not problems:
        logger.info("无需修复")
        return

    # 修复
    logger.info("\n开始修复时间戳...")
    backup = not args.no_backup
    fixed_count = fix_timestamps(dataset_dir, fps, target_episodes, backup=backup)

    logger.info(f"\n✅ 修复完成！共修改了 {fixed_count} 个 parquet 文件")

    # 验证修复结果
    logger.info("\n正在验证修复结果...")
    remaining_problems = check_timestamps(dataset_dir, fps, args.tolerance_s, target_episodes)
    if not remaining_problems:
        logger.info("✅ 验证通过，所有时间戳已修复")
    else:
        logger.warning(f"⚠️  仍有 {len(remaining_problems)} 个 episode 存在问题，请检查")


if __name__ == "__main__":
    main()
