#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot数据集数据分布提琴图绘制工具

功能说明：
    加载LeRobot格式的数据集，提取状态和动作数据，绘制提琴图展示数据分布

主要功能：
    1. 加载LeRobot数据集（从本地目录）
    2. 提取observation.state和action数据
    3. 绘制提琴图展示各维度的数据分布
    4. 支持保存图片到文件

配置参数：
    repo_id: LeRobot数据集的repo id（必需）
    root: 数据集根目录（可选）
    output_path: 输出图片保存路径（可选，默认保存到当前目录）
    max_episodes: 最大处理的episode数量（可选，默认处理全部）

使用示例：
    # 基本使用
    python plot_lerobot_distribution.py --repo_id lerobot/pusht
    
    # 指定本地root路径
    python plot_lerobot_distribution.py --repo_id lerobot/pusht --root ./lerobot_data
    
    # 指定输出路径
    python plot_lerobot_distribution.py --repo_id lerobot/pusht --output_dir ./output
    
    # 限制处理的episode数量（快速预览）
    python plot_lerobot_distribution.py --dataset_path ./lerobot_data --max_episodes 10

依赖要求：
    - lerobot库（pip install lerobot）
    - matplotlib, seaborn, numpy, pandas

作者：标准数据处理流程
日期：2024
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("错误: 请先安装lerobot库: pip install lerobot")
    exit(1)

import json


def load_feature_names(dataset_path: Path, feature_key: str = 'action'):
    """从info.json加载指定feature维度的名称"""
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        return None
    
    try:
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        if 'features' in info and feature_key in info['features']:
            feat_info = info['features'][feature_key]
            if 'names' in feat_info and feat_info['names']:
                # names是一个嵌套列表，取第一个元素
                names = feat_info['names'][0] if isinstance(feat_info['names'][0], list) else feat_info['names']
                return names
    except Exception as e:
        print(f"   ⚠️ 无法读取{feature_key}名称: {e}")
    
    return None


def load_lerobot_dataset(repo_id: str, root: str | None = None):
    """加载LeRobot数据集"""
    print(f"正在加载数据集: {repo_id}")
    if root:
        print(f"   - root: {root}")
    
    # 加载数据集
    try:
        dataset = LeRobotDataset(repo_id, root=root)
        print(f"✅ 数据集加载成功")
        print(f"   - 数据集路径: {dataset.root}")
        try:
            print(f"   - 总episode数: {len(dataset)}")
        except:
            print(f"   - 数据集已加载（无法获取episode数）")
    except Exception as e:
        raise ValueError(f"加载数据集失败: {e}\n请确保数据集路径正确且格式为LeRobot v2.0格式")
    
    return dataset


def extract_feature_fast(dataset: LeRobotDataset, start_idx: int, end_idx: int, feature_key: str):
    """高效提取指定feature数据，优先使用hf_dataset"""
    data_list = []
    
    try:
        # 优先使用hf_dataset直接访问列（最快）
        if hasattr(dataset, 'hf_dataset') and dataset.hf_dataset is not None:
            hf_ds = dataset.hf_dataset
            # 检查列是否存在
            if feature_key in hf_ds.column_names:
                feature_data = hf_ds.select(range(start_idx, end_idx))[feature_key]
                
                for item in feature_data:
                    if isinstance(item, torch.Tensor):
                        item = item.cpu().numpy()
                    elif isinstance(item, list):
                        item = np.array(item)
                    data_list.append(np.array(item).flatten())
                return data_list
    except Exception:
        pass
    
    # 回退到逐个样本加载
    for idx in range(start_idx, end_idx):
        try:
            sample = dataset[idx]
            if feature_key in sample:
                item = sample[feature_key]
                if isinstance(item, torch.Tensor):
                    item = item.cpu().numpy()
                data_list.append(item.flatten())
            else:
                # 尝试模糊匹配 (比如 observation.state 可能会被处理成 observation_state)
                # 但 LeRobotDataset __getitem__ 通常会处理好 key，或者 hf_dataset 列名是 observation.state
                # 这里作为一个 fallback
                keys = [k for k in sample.keys() if feature_key.replace('.', '_') in k]
                if keys:
                    item = sample[keys[0]]
                    if isinstance(item, torch.Tensor):
                        item = item.cpu().numpy()
                    data_list.append(item.flatten())
        except Exception:
            continue
    
    return data_list


def extract_data(dataset: LeRobotDataset, dataset_path: Path, max_episodes: int = None):
    """从数据集中提取动作和状态数据，并检查异常"""
    print("\nExtracting data...")
    
    features_to_extract = ['action', 'observation.state']
    feature_data = {k: [] for k in features_to_extract}
    feature_names = {}
    
    # 加载feature names
    for key in features_to_extract:
        names = load_feature_names(dataset_path, key)
        if names:
            print(f"   - Found {key} dimension names: {len(names)} dimensions")
            feature_names[key] = names
        else:
            print(f"   - {key} dimension names not found")
            feature_names[key] = None

    episode_lengths = []
    
    # 异常统计
    abnormal_episodes = set()
    total_episodes_processed = 0
    
    # 获取数据集的总长度和episode信息
    total_samples = None
    num_episodes = None
    
    try:
        total_samples = len(dataset)
        if hasattr(dataset, 'episode_data_index') and dataset.episode_data_index is not None:
            episode_index = dataset.episode_data_index
            # 检查是否是 "from"/"to" 格式
            if isinstance(episode_index, dict) and 'from' in episode_index and 'to' in episode_index:
                from_indices = episode_index['from']
                if isinstance(from_indices, torch.Tensor):
                    num_episodes = len(from_indices)
                else:
                    num_episodes = len(from_indices)
            elif isinstance(episode_index, dict):
                num_episodes = len([k for k in episode_index.keys() if k != 'from' and k != 'to'])
            elif isinstance(episode_index, (list, tuple)):
                num_episodes = len(episode_index)
            else:
                num_episodes = None
        else:
            num_episodes = None
        
        if num_episodes is None:
            print(f"   - Total samples: {total_samples}")
            print(f"   - Episode count: unknown")
        else:
            print(f"   - Total samples: {total_samples}")
            print(f"   - Episode count: {num_episodes}")
            if max_episodes is not None:
                num_episodes = min(num_episodes, max_episodes)
                print(f"   - Will process first {num_episodes} episodes (limited by --max_episodes)")
    except Exception as e:
        num_episodes = max_episodes if max_episodes else None
        print(f"   - Warning: Could not get dataset info: {e}")
        if num_episodes:
            print(f"   - Will process first {num_episodes} episodes")
    
    # 方法1: 如果有episode_data_index，按episode提取
    has_episode_index = (hasattr(dataset, 'episode_data_index') and 
                         dataset.episode_data_index is not None and
                         len(dataset.episode_data_index) > 0)
    
    if has_episode_index:
        episode_index = dataset.episode_data_index
        is_dict = isinstance(episode_index, dict)
        
        # 统一获取episode的start/end索引
        episodes_indices = []
        
        if is_dict and 'from' in episode_index and 'to' in episode_index:
            from_indices = episode_index['from']
            to_indices = episode_index['to']
            if isinstance(from_indices, torch.Tensor):
                from_indices = from_indices.cpu().numpy()
            if isinstance(to_indices, torch.Tensor):
                to_indices = to_indices.cpu().numpy()
            
            # flatten
            if len(from_indices.shape) > 1: from_indices = from_indices.flatten()
            if len(to_indices.shape) > 1: to_indices = to_indices.flatten()
            
            count = min(len(from_indices), len(to_indices))
            if num_episodes is not None: count = min(count, num_episodes)
            
            for i in range(count):
                episodes_indices.append((i, int(from_indices[i]), int(to_indices[i])))
                
        elif is_dict:
            keys = sorted([k for k in episode_index.keys() if k != 'from' and k != 'to'])
            if num_episodes: keys = keys[:num_episodes]
            
            for k in keys:
                info = episode_index[k]
                start_idx, end_idx = 0, 0
                if isinstance(info, dict):
                    start_idx = info.get('start', info.get('start_idx', 0))
                    end_idx = info.get('end', info.get('end_idx', start_idx + 1))
                elif isinstance(info, (list, tuple)) and len(info) >= 2:
                    start_idx, end_idx = info[0], info[1]
                elif isinstance(info, torch.Tensor):
                    if info.numel() >= 2:
                        start_idx, end_idx = info[0].item(), info[1].item()
                    else:
                        start_idx = info.item()
                        end_idx = start_idx + 1
                episodes_indices.append((k, int(start_idx), int(end_idx)))
        
        total_episodes_processed = len(episodes_indices)
        
        # 遍历处理
        for ep_id, start_idx, end_idx in tqdm(episodes_indices, desc="处理episode"):
            if start_idx >= end_idx: continue
            
            episode_length = end_idx - start_idx
            episode_lengths.append(episode_length)
            
            # 提取每个feature
            is_abnormal = False
            for key in features_to_extract:
                data = extract_feature_fast(dataset, start_idx, end_idx, key)
                if data:
                    np_data = np.array(data)
                    feature_data[key].append(np_data)
                    
                    # 检查异常值 (只针对 observation.state 和 action)
                    # 阈值: +/- PI
                    limit = np.pi
                    if np.any(np.abs(np_data) > limit):
                        is_abnormal = True
            
            if is_abnormal:
                abnormal_episodes.add(ep_id)

    else:
        # 方法2: 样本级提取（不支持精确的episode统计和异常ID追踪，只能粗略估计）
        print("   使用样本级提取模式...")
        # ... (简化处理，暂时略过复杂的样本级异常ID追踪，仅提取数据)
        # 为保持代码简洁，建议用户使用较新版本的LeRobot数据集格式
        pass
    
    # 合并数据
    final_data = {}
    for key, data_list in feature_data.items():
        if data_list:
            try:
                # 检查维度一致性
                shapes = [d.shape[1] if len(d.shape) > 1 else len(d) for d in data_list]
                if len(set(shapes)) == 1:
                    final_data[key] = np.concatenate(data_list, axis=0)
                else:
                    print(f"⚠️ {key} 维度不一致, 跳过")
                    final_data[key] = None
            except Exception as e:
                print(f"⚠️ 合并 {key} 数据出错: {e}")
                final_data[key] = None
        else:
            final_data[key] = None
            
    return final_data, episode_lengths, feature_names, abnormal_episodes, total_episodes_processed


def plot_violin_distribution(data: np.ndarray, feature_names: list, title: str, output_path: Path = None):
    """绘制提琴图"""
    if data is None or data.size == 0:
        print(f"⚠️ 没有{title}数据可绘制")
        return
    
    num_features = data.shape[1]
    
    # 如果特征太多，只绘制前20个
    if num_features > 20:
        print(f"⚠️ Too many features ({num_features}), only plotting first 20 features")
        data = data[:, :20]
        feature_names = feature_names[:20]
        num_features = 20
    
    # 设置绘图风格
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
    
    # 创建子图，每个特征一个子图，每个子图有自己的y轴范围
    n_cols = min(4, num_features)  # 每行最多4个子图
    n_rows = (num_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    fig.suptitle(f'{title} - Distribution Violin Plot', fontsize=16, fontweight='bold', y=0.995)
    
    # 确保axes是二维数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        feature_data = data[:, i]
        feature_name = feature_names[i] if i < len(feature_names) else f"Feature{i+1}"
        
        # 为每个特征创建单独的DataFrame
        df = pd.DataFrame({'Value': feature_data})
        
        # 绘制单个特征的提琴图
        sns.violinplot(data=df, y='Value', ax=ax, inner='box')
        
        # 设置标题和标签
        ax.set_title(feature_name, fontsize=10, fontweight='bold')
        ax.set_ylabel('Value', fontsize=9)
        ax.set_xlabel('', fontsize=0)  # 移除x轴标签
        
        # 每个子图使用自己的y轴范围
        data_min = feature_data.min()
        data_max = feature_data.max()
        data_range = data_max - data_min
        margin = data_range * 0.1 if data_range > 0 else 0.1
        ax.set_ylim(data_min - margin, data_max + margin)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(num_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_episode_length_distribution(episode_lengths: list, output_path: Path = None):
    """绘制episode长度分布"""
    if not episode_lengths:
        print("⚠️ No episode length data to plot")
        return
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制提琴图
    df = pd.DataFrame({'Episode Length': episode_lengths})
    sns.violinplot(data=df, y='Episode Length', ax=ax)
    
    ax.set_title('Episode Length Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Frames', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Episode length distribution plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_feature_names(num_features: int, prefix: str = "Feature") -> list:
    """生成特征名称列表"""
    return [f"{prefix}{i+1}" for i in range(num_features)]


def main():
    parser = argparse.ArgumentParser(
        description='LeRobot数据集数据分布提琴图绘制工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--repo_id', type=str, required=True,
                       help='LeRobot数据集的repo id（必需，例如：lerobot/pusht）')
    parser.add_argument('--root', type=str, default=None,
                       help='数据集根目录（可选，默认：~/.cache/huggingface/lerobot）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出图片保存目录（默认：当前目录）')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='最大处理的episode数量（默认：处理全部）')
    parser.add_argument('--prefix', type=str, default='distribution',
                       help='输出文件名前缀（默认：distribution）')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()
    
    # 加载数据集
    dataset = load_lerobot_dataset(args.repo_id, args.root)
    dataset_path = dataset.root
    
    # 提取数据
    feature_data_dict, episode_lengths, feature_names_dict, abnormal_episodes, total_episodes = extract_data(dataset, dataset_path, args.max_episodes)
    
    # 打印异常统计
    print(f"\n📊 异常数据检测报告 (阈值: ±π):")
    if total_episodes > 0:
        abnormal_count = len(abnormal_episodes)
        abnormal_ratio = abnormal_count / total_episodes * 100
        print(f"   - 总episode数: {total_episodes}")
        print(f"   - 异常episode数: {abnormal_count}")
        print(f"   - 异常比例: {abnormal_ratio:.2f}%")
        if abnormal_count > 0:
            # 如果异常太多，只打印前50个
            abnormal_list = sorted(list(abnormal_episodes))
            print(f"   - 异常episode ID列表: {abnormal_list[:50]}")
            if len(abnormal_list) > 50:
                print(f"     ... (共 {len(abnormal_list)} 个，仅显示前50个)")
    else:
        print("   - 未处理任何episode，无法统计异常")

    # 绘制各特征分布
    has_data = False
    
    for feature_key, data_array in feature_data_dict.items():
        if data_array is None or data_array.size == 0 or len(data_array.shape) != 2:
            print(f"⚠️ Skipping {feature_key} plot (no valid data)")
            continue
            
        print(f"\n正在绘制 {feature_key} 分布...")
        num_dims = data_array.shape[1]
        names = feature_names_dict.get(feature_key)
        
        # 简化名称以便在图表中显示，并分组
        if names and len(names) == num_dims:
            left_indices = []
            right_indices = []
            left_names = []
            right_names = []
            
            for idx, name in enumerate(names):
                # 简化名称：提取关键部分
                parts = name.split('.')
                if len(parts) >= 3:
                    # 取最后两部分
                    simplified = '_'.join(parts[-2:])
                else:
                    simplified = name.replace('.', '_')
                
                # 根据名称判断是左还是右
                if 'masterLeft' in name or 'Left' in name or 'pikaSensor_l' in name or 'pikaGripper_l' in name:
                    left_indices.append(idx)
                    left_names.append(simplified)
                elif 'masterRight' in name or 'Right' in name or 'pikaSensor_r' in name or 'pikaGripper_r' in name:
                    right_indices.append(idx)
                    right_names.append(simplified)
                else:
                    # 如果无法判断，默认归为左侧（前14维通常是左侧）
                    if idx < num_dims // 2:
                        left_indices.append(idx)
                        left_names.append(simplified)
                    else:
                        right_indices.append(idx)
                        right_names.append(simplified)
            
            print(f"   - Using dimension names from info.json")
        else:
            # 生成默认名称，按维度数平分
            mid_point = num_dims // 2
            left_indices = list(range(mid_point))
            right_indices = list(range(mid_point, num_dims))
            left_names = generate_feature_names(len(left_indices), "Left")
            right_names = generate_feature_names(len(right_indices), "Right")
            if names:
                print(f"   ⚠️ 名称数量({len(names)})与维度数({num_dims})不匹配，使用默认名称")
        
        # 绘制左侧数据
        clean_key = feature_key.replace('.', '_')
        if left_indices:
            left_data = data_array[:, left_indices]
            left_output_path = output_dir / f"{args.prefix}_{clean_key}_left.png"
            plot_violin_distribution(left_data, left_names, f"{feature_key} - Left Arm Distribution", left_output_path)
            has_data = True
        
        # 绘制右侧数据
        if right_indices:
            right_data = data_array[:, right_indices]
            right_output_path = output_dir / f"{args.prefix}_{clean_key}_right.png"
            plot_violin_distribution(right_data, right_names, f"{feature_key} - Right Arm Distribution", right_output_path)
            has_data = True

    # 绘制episode长度分布
    if episode_lengths and len(episode_lengths) > 0:
        length_output_path = output_dir / f"{args.prefix}_episode_lengths.png"
        plot_episode_length_distribution(episode_lengths, length_output_path)
        has_data = True
    else:
        print("⚠️ 跳过episode长度分布图（无有效数据）")
    
    if has_data:
        print("\n✅ 所有图表绘制完成！")
    else:
        print("\n❌ 未找到可绘制的数据，请检查数据集格式是否正确")


if __name__ == "__main__":
    main()
