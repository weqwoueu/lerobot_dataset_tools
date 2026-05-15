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
import sys
from pathlib import Path
from typing import Set

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

    # 尝试作为文件路径读取
    file_path = Path(episodes_arg)
    if file_path.exists() and file_path.is_file():
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
    parts = episodes_arg.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            # 支持范围，如 10-20
            try:
                start, end = map(int, part.split('-'))
                episodes_to_delete.update(range(start, end + 1))
            except ValueError:
                logger.warning(f"无法解析范围: {part}")
        elif part.isdigit():
            episodes_to_delete.add(int(part))
        else:
            logger.warning(f"忽略无效的 episode ID: {part}")
            
    return episodes_to_delete

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
    
    logger.info(f"统计:")
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
        local_output_dir=output_dir
    )
    
    if success:
        logger.info(f"✅ 数据集处理完成! 输出目录: {output_dir}")
    else:
        logger.error("❌ 数据集创建失败")

if __name__ == "__main__":
    main()
