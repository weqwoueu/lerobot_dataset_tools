#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Darwin02 数据集数据分布提琴图绘制工具

针对 darwin02 机器人数据集，按 5 个物理类别分别绘制数据分布提琴图，
输出为统一的多页 PDF 文件。

# 0410 版数据集
5 个类别， 共43维：
    1. 左臂 (Left Arm): 7个关节 + 2个吸盘 + 6维力/力矩  (15维)
    2. 右臂 (Right Arm): 7个关节 + 2个吸盘 + 6维力/力矩 (15维)
    3. 躯干关节 (Trunk): head, lifter, neck, waist        (4维)
    4. 底盘 (Chassis): 位置 xyz + 四元数姿态 xyzw         (7维)
    5. 控制标志位 (Control Flags): 2个                     (2维)

# 0501、0502 版数据集
5 个类别，共45维：
    1. 左臂 (Left Arm): 7个关节 + 3个吸盘（控制量+负压检测IO+红外IO） + 6维力/力矩  (16维)
    2. 右臂 (Right Arm): 7个关节 + 3个吸盘（控制量+负压检测IO+红外IO） + 6维力/力矩 (16维)
    3. 躯干关节 (Trunk): head, lifter, neck, waist        (4维)
    4. 底盘 (Chassis): 位置 xyz + 四元数姿态 xyzw         (7维)
    5. 控制标志位 (Control Flags): 2个                     (2维)

使用示例：
    python 0_plot_darwin02_distribution.py \
        --repo_id standard/darwin02_0401_lerobot \
        --root ./.cache/huggingface/lerobot \
        --output_dir ./output \
        --max_episodes 5

    # .cache/huggingface/lerobot/standard/darwin02_0501_2
    # .cache/huggingface/lerobot/standard/darwin02_0502
    # .cache/huggingface/lerobot/standard/darwin02_0502_error
    python 0_plot_darwin02_distribution.py \
        --repo_id standard/darwin02_0501_2 \
        --root /home/standard/workspace/gitlab/openpi/.cache/huggingface/lerobot/standard/darwin02_0501_2 \
        --output_dir ./output
        # --max_episodes 5
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

_FONT_DIR = Path(__file__).resolve().parent
_SC_FONT = _FONT_DIR / "fonts/NotoSansCJK-SC-Regular.otf"
if _SC_FONT.exists():
    fm.fontManager.addfont(str(_SC_FONT))
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("错误: 请先安装lerobot库: pip install lerobot")
    exit(1)


# ──────────────────────────────────────────────
# Darwin02 维度分类规则
# ──────────────────────────────────────────────
COLOR_NORMAL = "#5B9BD5"
COLOR_ABNORMAL = "#CD5C5C"

CATEGORY_RULES = [
    {
        "name": "Left Arm",
        "name_cn": "左臂",
        "description": "7关节(0-6) + 吸盘开关/负压(7-8) + 六维力(9-14)  共15维",
        "match": lambda n: n.startswith("left_"),
    },
    {
        "name": "Right Arm",
        "name_cn": "右臂",
        "description": "7关节(15-21) + 吸盘开关/负压(22-23) + 六维力(24-29)  共15维",
        "match": lambda n: n.startswith("right_"),
    },
    {
        "name": "Trunk Joints",
        "name_cn": "头部腰部关节",
        "description": "头部俯仰(30) / 腰部升降(31) / 脖子航向(32) / 腰部俯仰(33)  共4维",
        "match": lambda n: n in ("head_joint", "lifter_joint", "neck_joint", "waist_joint"),
    },
    {
        "name": "Chassis Pose",
        "name_cn": "底盘位姿",
        "description": "位置 xyz(34-36) + 四元数姿态 xyzw(37-40)  共7维",
        "match": lambda n: n.startswith("chassis.pose."),
    },
    {
        "name": "Control Flags",
        "name_cn": "控制信号",
        "description": "遥控使能(41) / OBA使能(42)  共2维",
        "match": lambda n: n.startswith("chassis.remote_"),
    },
]


def _is_joint(name: str) -> bool:
    return name.endswith("_joint")


def _is_joint_abnormal(col: np.ndarray) -> bool:
    """关节角数据是否存在超出 [-pi, pi] 的异常值。"""
    return bool(np.any(np.abs(col) > np.pi))


def classify_dimensions(names: list[str]) -> list[dict]:
    """根据 CATEGORY_RULES 将维度名列表分到各类别中。"""
    assigned: set[int] = set()
    categories = []

    for rule in CATEGORY_RULES:
        indices, dim_names = [], []
        for idx, name in enumerate(names):
            if idx not in assigned and rule["match"](name):
                indices.append(idx)
                dim_names.append(name)
                assigned.add(idx)
        if indices:
            categories.append(
                {
                    "name": rule["name"],
                    "name_cn": rule["name_cn"],
                    "description": rule["description"],
                    "indices": indices,
                    "dim_names": dim_names,
                }
            )

    remaining = [i for i in range(len(names)) if i not in assigned]
    if remaining:
        categories.append(
            {
                "name": "Other",
                "name_cn": "其他",
                "description": f"{len(remaining)} unclassified dims",
                "indices": remaining,
                "dim_names": [names[i] for i in remaining],
            }
        )

    return categories


# ──────────────────────────────────────────────
# 数据加载 / 提取（复用自 0_plot_lerobot_distribution.py）
# ──────────────────────────────────────────────


def load_feature_names(dataset_path: Path, feature_key: str = "action"):
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        return None
    try:
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        if "features" in info and feature_key in info["features"]:
            feat_info = info["features"][feature_key]
            if "names" in feat_info and feat_info["names"]:
                raw = feat_info["names"]
                return raw[0] if isinstance(raw[0], list) else raw
    except Exception as e:
        print(f"   ⚠️ 无法读取 {feature_key} 名称: {e}")
    return None


def load_lerobot_dataset(repo_id: str, root: str | None = None):
    print(f"正在加载数据集: {repo_id}")
    if root:
        print(f"   root: {root}")
    try:
        dataset = LeRobotDataset(repo_id, root=root)
        print(f"✅ 数据集加载成功  路径: {dataset.root}")
        try:
            print(f"   总样本数: {len(dataset)}")
        except Exception:
            pass
    except Exception as e:
        raise ValueError(f"加载数据集失败: {e}")
    return dataset


def _extract_feature_fast(dataset, start_idx: int, end_idx: int, feature_key: str):
    data_list: list[np.ndarray] = []
    try:
        if hasattr(dataset, "hf_dataset") and dataset.hf_dataset is not None:
            hf_ds = dataset.hf_dataset
            if feature_key in hf_ds.column_names:
                for item in hf_ds.select(range(start_idx, end_idx))[feature_key]:
                    if isinstance(item, torch.Tensor):
                        item = item.cpu().numpy()
                    elif isinstance(item, list):
                        item = np.array(item)
                    data_list.append(np.asarray(item).flatten())
                return data_list
    except Exception:
        pass

    for idx in range(start_idx, end_idx):
        try:
            sample = dataset[idx]
            if feature_key in sample:
                v = sample[feature_key]
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                data_list.append(v.flatten())
        except Exception:
            continue
    return data_list


def _parse_episode_indices(dataset, max_episodes: int | None):
    """统一解析 episode_data_index，返回 [(ep_id, start, end), ...]。"""
    if not (
        hasattr(dataset, "episode_data_index")
        and dataset.episode_data_index is not None
        and len(dataset.episode_data_index) > 0
    ):
        return []

    ep_idx = dataset.episode_data_index
    result = []

    if isinstance(ep_idx, dict) and "from" in ep_idx and "to" in ep_idx:
        f = ep_idx["from"]
        t = ep_idx["to"]
        if isinstance(f, torch.Tensor):
            f = f.cpu().numpy().flatten()
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy().flatten()
        count = min(len(f), len(t))
        if max_episodes is not None:
            count = min(count, max_episodes)
        for i in range(count):
            result.append((i, int(f[i]), int(t[i])))
    elif isinstance(ep_idx, dict):
        keys = sorted(k for k in ep_idx if k not in ("from", "to"))
        if max_episodes:
            keys = keys[:max_episodes]
        for k in keys:
            info = ep_idx[k]
            if isinstance(info, dict):
                s = info.get("start", info.get("start_idx", 0))
                e = info.get("end", info.get("end_idx", s + 1))
            elif isinstance(info, (list, tuple)) and len(info) >= 2:
                s, e = info[0], info[1]
            elif isinstance(info, torch.Tensor):
                s = info[0].item() if info.numel() >= 2 else info.item()
                e = info[1].item() if info.numel() >= 2 else s + 1
            else:
                continue
            result.append((k, int(s), int(e)))

    return result


def extract_data(dataset, dataset_path: Path, max_episodes: int | None = None):
    """提取 action / observation.state 数据，返回合并后的数组字典。"""
    print("\n正在提取数据...")
    keys = ["action", "observation.state"]
    feature_data: dict[str, list] = {k: [] for k in keys}
    feature_names: dict[str, list | None] = {}
    episode_lengths: list[int] = []

    for k in keys:
        names = load_feature_names(dataset_path, k)
        feature_names[k] = names
        if names:
            print(f"   {k}: {len(names)} 维")

    episodes = _parse_episode_indices(dataset, max_episodes)
    if not episodes:
        print("   ⚠️ 无法获取 episode 索引，跳过提取")
        return {k: None for k in keys}, episode_lengths, feature_names

    print(f"   将处理 {len(episodes)} 个 episode")
    for _ep_id, start, end in tqdm(episodes, desc="处理 episode"):
        if start >= end:
            continue
        episode_lengths.append(end - start)
        for k in keys:
            data = _extract_feature_fast(dataset, start, end, k)
            if data:
                feature_data[k].append(np.array(data))

    final: dict[str, np.ndarray | None] = {}
    for k, chunks in feature_data.items():
        if not chunks:
            final[k] = None
            continue
        dims = {c.shape[1] for c in chunks if c.ndim == 2}
        if len(dims) == 1:
            final[k] = np.concatenate(chunks, axis=0)
        else:
            print(f"   ⚠️ {k} 各 episode 维度不一致，跳过")
            final[k] = None

    return final, episode_lengths, feature_names


# ──────────────────────────────────────────────
# PDF 绘图
# ──────────────────────────────────────────────


def _set_plot_style():
    for style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "default"):
        try:
            plt.style.use(style)
            break
        except Exception:
            continue
    if _SC_FONT.exists():
        fm.fontManager.addfont(str(_SC_FONT))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def _is_discrete(col: np.ndarray, max_unique: int = 5) -> bool:
    """判断一列数据是否为离散值（unique 值数量 <= max_unique）。"""
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
    ax.set_ylim(0, max(ratios) * 1.25)


def _plot_category_page(
    pdf: PdfPages,
    cat_data: np.ndarray,
    dim_names: list[str],
    page_title: str,
) -> list[tuple[str, float, float, bool]]:
    """向 PDF 写入一页。
    返回 [(dim_name, min, max, is_joint), ...] 供打印。
    关节维度超出 [-pi, pi] 的用红色标记；
    离散数据（unique <= 5）用柱状占比图代替提琴图。
    """
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
        col = cat_data[:, i]
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
    """在 stdout 打印一个类别内各维度的 y 轴范围表格。
    关节维度额外显示角度值，超限的标记 [!]。
    """
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


def create_pdf_report(feature_data_dict: dict, feature_names_dict: dict, episode_lengths: list[int], output_path: Path):
    _set_plot_style()

    with warnings.catch_warnings(), PdfPages(output_path) as pdf:
        warnings.simplefilter("ignore", UserWarning)

        for fkey, arr in feature_data_dict.items():
            if arr is None or arr.size == 0 or arr.ndim != 2:
                print(f"  [WARN] skip {fkey} (no valid data)")
                continue

            names = feature_names_dict.get(fkey)
            if names and len(names) == arr.shape[1]:
                categories = classify_dimensions(names)
            else:
                categories = [
                    {
                        "name": "All",
                        "name_cn": "全部",
                        "description": f"{arr.shape[1]} dims",
                        "indices": list(range(arr.shape[1])),
                        "dim_names": [f"dim_{i}" for i in range(arr.shape[1])],
                    }
                ]

            for cat in categories:
                idx = cat["indices"]
                if not idx:
                    continue
                title = f"{fkey}  ·  {cat['name']} ({cat['name_cn']})\n{cat['description']}"
                ranges = _plot_category_page(
                    pdf,
                    arr[:, idx],
                    cat["dim_names"],
                    title,
                )
                header = f"{fkey}  ·  {cat['name']} ({cat['name_cn']})"
                _print_ranges_table(header, ranges)

        if episode_lengths:
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

    print(f"\nPDF saved: {output_path}")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Darwin02 数据集分布提琴图（统一 PDF 输出）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot 数据集 repo id")
    parser.add_argument("--root", type=str, default=None, help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认当前目录）")
    parser.add_argument("--max_episodes", type=int, default=None, help="最大处理 episode 数量")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_lerobot_dataset(args.repo_id, args.root)
    dataset_path = dataset.root

    dataset_name = args.repo_id.replace("/", "_")
    if args.root:
        candidate = Path(args.root) / args.repo_id
        if candidate.is_dir():
            dataset_name = candidate.resolve().name
        else:
            dataset_name = Path(args.root).resolve().name

    feature_data_dict, episode_lengths, feature_names_dict = extract_data(
        dataset,
        dataset_path,
        args.max_episodes,
    )

    output_path = output_dir / f"{dataset_name}_distribution.pdf"
    print("\n正在生成 PDF 报告...")
    create_pdf_report(feature_data_dict, feature_names_dict, episode_lengths, output_path)


if __name__ == "__main__":
    main()
