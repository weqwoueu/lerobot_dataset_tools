#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot 数据集数据分布 PDF 报告工具

加载 LeRobot 格式数据集，提取 action 和 observation.state，按原始维度顺序
分页绘制分布图，并输出为一个统一 PDF 文件。action 维度不会按左右臂或名称
重排，也不会截断。

使用示例：
    python 0_plot_lerobot_distribution.py \
        --repo_id lerobot/pusht \
        --root ./lerobot_data \
        --output_dir ./output \
        --max_episodes 10
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# 设置matplotlib支持中文
_FONT_DIR = Path(__file__).resolve().parent
_SC_FONT = _FONT_DIR / "fonts/NotoSansCJK-SC-Regular.otf"
if _SC_FONT.exists():
    fm.fontManager.addfont(str(_SC_FONT))
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans", "Arial", "SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42

COLOR_NORMAL = "#5B9BD5"
COLOR_ABNORMAL = "#CD5C5C"
MAX_DIMS_PER_PAGE = 20

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("错误: 请先安装lerobot库: pip install lerobot")
    exit(1)


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


def _set_plot_style():
    """设置 PDF 报告的统一绘图风格。"""
    for style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "default"):
        try:
            plt.style.use(style)
            break
        except Exception:
            continue
    if _SC_FONT.exists():
        fm.fontManager.addfont(str(_SC_FONT))
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "DejaVu Sans",
        "Arial",
        "SimHei",
        "WenQuanYi Micro Hei",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def _is_joint(name: str) -> bool:
    return name.endswith("_joint")


def _is_joint_abnormal(col: np.ndarray) -> bool:
    """关节角数据是否存在超出 [-pi, pi] 的异常值。"""
    return bool(np.any(np.abs(col) > np.pi))


def _is_discrete(col: np.ndarray, max_unique: int = 5) -> bool:
    """判断一列数据是否为离散值。"""
    return len(np.unique(col)) <= max_unique


def _plot_discrete(ax, col: np.ndarray, color: str):
    """为离散数据绘制柱状占比图。"""
    unique_vals, counts = np.unique(col, return_counts=True)
    ratios = counts / len(col) * 100
    bars = ax.bar([str(v) for v in unique_vals], ratios, color=color, edgecolor="white", width=0.5)
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{ratio:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylabel("%", fontsize=8)
    ax.set_ylim(0, max(ratios) * 1.25 if len(ratios) else 1)


def _plot_category_page(
    pdf: PdfPages,
    page_data: np.ndarray,
    dim_names: list[str],
    page_title: str,
) -> list[tuple[str, float, float, bool]]:
    """向 PDF 写入一页，维度顺序与 page_data 列顺序完全一致。"""
    n = len(dim_names)
    n_cols = min(5, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 3.8))
    fig.suptitle(page_title, fontsize=13, fontweight="bold", y=0.995)

    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.atleast_2d(axes)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    ranges: list[tuple[str, float, float, bool]] = []

    for i in range(n):
        ax = axes[i // n_cols, i % n_cols]
        col = page_data[:, i]
        name = dim_names[i]
        joint = _is_joint(name)
        abnormal = joint and _is_joint_abnormal(col)
        color = COLOR_ABNORMAL if abnormal else COLOR_NORMAL
        discrete = _is_discrete(col)

        d_min, d_max = float(col.min()), float(col.max())
        ranges.append((name, d_min, d_max, joint))

        if discrete:
            _plot_discrete(ax, col, color)
            unique_str = ", ".join(f"{v}" for v in np.unique(col))
            subtitle = f"{name}\nvals: {{{unique_str}}}"
        elif joint:
            sns.violinplot(y=col, ax=ax, inner="box", color=color)
            margin = (d_max - d_min) * 0.1 if d_max > d_min else 0.1
            ax.set_ylim(d_min - margin, d_max + margin)
            deg_min, deg_max = np.degrees(d_min), np.degrees(d_max)
            subtitle = f"{name}\n[{d_min:.3f}, {d_max:.3f}] rad\n[{deg_min:.1f}, {deg_max:.1f}] deg"
        else:
            sns.violinplot(y=col, ax=ax, inner="box", color=color)
            margin = (d_max - d_min) * 0.1 if d_max > d_min else 0.1
            ax.set_ylim(d_min - margin, d_max + margin)
            subtitle = f"{name}\n[{d_min:.3f}, {d_max:.3f}]"

        title_color = COLOR_ABNORMAL if abnormal else "black"
        ax.set_title(subtitle, fontsize=7, fontweight="bold", color=title_color)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.3)

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=200)
    plt.close(fig)
    return ranges


def _print_ranges_table(header: str, ranges: list[tuple[str, float, float, bool]]):
    """在 stdout 打印一页内各维度的范围。"""
    if not ranges:
        return

    name_w = max(len(r[0]) for r in ranges)
    name_w = max(name_w, 3)
    has_joints = any(r[3] for r in ranges)

    print(f"\n{header}")
    if has_joints:
        print(
            f"  {'dim':<{name_w}}  {'min(rad)':>12}  {'max(rad)':>12}"
            f"  {'min(deg)':>10}  {'max(deg)':>10}  {'status':>6}"
        )
    else:
        print(f"  {'dim':<{name_w}}  {'min':>12}  {'max':>12}")

    for name, lo, hi, joint in ranges:
        if joint:
            deg_lo, deg_hi = np.degrees(lo), np.degrees(hi)
            flag = " [!]" if (abs(lo) > np.pi or abs(hi) > np.pi) else "  ok"
            print(f"  {name:<{name_w}}  {lo:>12.4f}  {hi:>12.4f}  {deg_lo:>10.1f}  {deg_hi:>10.1f}  {flag:>6}")
        else:
            print(f"  {name:<{name_w}}  {lo:>12.4f}  {hi:>12.4f}")


def _feature_names_for_plot(names: list | None, num_dims: int, feature_key: str) -> list[str]:
    """返回绘图名称；不改变原始维度顺序。"""
    if names and len(names) == num_dims:
        print(f"   - {feature_key} 使用 info.json 中的 {num_dims} 个维度名")
        return [str(name) for name in names]
    if names:
        print(f"   ⚠️ {feature_key} 名称数量({len(names)})与维度数({num_dims})不匹配，使用 dim_N 名称")
    else:
        print(f"   - {feature_key} 未找到维度名，使用 dim_N 名称")
    return [f"dim_{i}" for i in range(num_dims)]


def create_pdf_report(
    feature_data_dict: dict,
    feature_names_dict: dict,
    episode_lengths: list[int],
    output_path: Path,
    max_dims_per_page: int = MAX_DIMS_PER_PAGE,
) -> bool:
    """生成统一 PDF 报告。返回是否写入了有效内容。"""
    _set_plot_style()
    has_data = False

    with warnings.catch_warnings(), PdfPages(output_path) as pdf:
        warnings.simplefilter("ignore", UserWarning)

        for feature_key, arr in feature_data_dict.items():
            if arr is None or arr.size == 0 or arr.ndim != 2:
                print(f"⚠️ 跳过 {feature_key}（无有效数据）")
                continue

            has_data = True
            num_dims = arr.shape[1]
            dim_names = _feature_names_for_plot(feature_names_dict.get(feature_key), num_dims, feature_key)
            print(f"\n正在写入 {feature_key} 分布：{num_dims} 维，原始顺序分页")

            for start in range(0, num_dims, max_dims_per_page):
                end = min(start + max_dims_per_page, num_dims)
                page_title = f"{feature_key} Distribution · dims {start}-{end - 1} / {num_dims}"
                ranges = _plot_category_page(
                    pdf,
                    arr[:, start:end],
                    dim_names[start:end],
                    page_title,
                )
                _print_ranges_table(f"{feature_key} · dims {start}-{end - 1}", ranges)

        if episode_lengths:
            has_data = True
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.violinplot(y=episode_lengths, ax=ax, inner="box")
            ax.set_title("Episode Length Distribution", fontsize=14, fontweight="bold")
            ax.set_ylabel("Frames")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig, dpi=200)
            plt.close(fig)
            ep_min, ep_max = min(episode_lengths), max(episode_lengths)
            print(f"\nEpisode Length: min={ep_min}, max={ep_max}, count={len(episode_lengths)}")
        else:
            print("⚠️ 跳过 episode 长度分布图（无有效数据）")

    if has_data:
        print(f"\n✅ PDF saved: {output_path}")
    return has_data


def main():
    parser = argparse.ArgumentParser(
        description='LeRobot 数据集数据分布 PDF 报告工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--repo_id', type=str, required=True,
                       help='LeRobot数据集的repo id（必需，例如：lerobot/pusht）')
    parser.add_argument('--root', type=str, default=None,
                       help='数据集根目录（可选，默认：~/.cache/huggingface/lerobot）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出 PDF 保存目录（默认：当前目录）')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='最大处理的episode数量（默认：处理全部）')
    parser.add_argument('--prefix', type=str, default='distribution',
                       help='输出 PDF 文件名前缀（默认：distribution）')
    
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

    dataset_name = args.repo_id.replace("/", "_")
    if args.root:
        candidate = Path(args.root) / args.repo_id
        if candidate.is_dir():
            dataset_name = candidate.resolve().name
        else:
            dataset_name = Path(args.root).resolve().name

    output_path = output_dir / f"{args.prefix}_{dataset_name}_distribution.pdf"
    print("\n正在生成 PDF 报告...")
    has_data = create_pdf_report(feature_data_dict, feature_names_dict, episode_lengths, output_path)

    if has_data:
        print("\n✅ 所有图表绘制完成！")
    else:
        print("\n❌ 未找到可绘制的数据，请检查数据集格式是否正确")


if __name__ == "__main__":
    main()
