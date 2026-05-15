#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Darwin02 数据集中的力传感器数据转换为 episode-start delta 模式。

原始数据包含较大的静态重力偏置（左臂 force.z 约 -79~-36，右臂约 -47~+6），
且左右臂区间相差悬殊。转换后每帧存储 force[t] - force[0]，消除静态偏置，
使分布中心化在 0 附近，便于模型学习。

转换的维度按 meta/info.json 中的字段名自动解析，兼容 43 维旧布局和 45 维新布局：
  left_wrench.force.{x,y,z}
  right_wrench.force.{x,y,z}

默认同时转换 observation.state 和 action 两列，且都使用 episode 首帧 observation.state force
作为 offset，使训练前处理里的 state/action force 目标保持在同一个 delta 坐标系。

使用示例：
    # in-place 修改（覆盖原数据集）
    python 6_convert_force_to_delta.py \\
        --repo_id standard/darwin02_0501_2 \\
        --root ./.cache/huggingface/lerobot/standard/darwin02_0501_2

    # 保存到新目录（不修改原数据集）
    python 6_convert_force_to_delta.py \\
        --repo_id standard/darwin02_0501_2 \\
        --root ./.cache/huggingface/lerobot/standard/darwin02_0501_2 \\
        --output_dir ./.cache/huggingface/lerobot/standard/darwin02_0501_2_force_delta

    # 仅预览，不写入
    python 6_convert_force_to_delta.py \\
        --repo_id standard/darwin02_0501_2 \\
        --root ./.cache/huggingface/lerobot/standard/darwin02_0501_2 \\
        --dry_run
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("错误: 请先安装 lerobot 库: pip install lerobot")
    raise

STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
DEFAULT_COLUMNS = (STATE_COLUMN, ACTION_COLUMN)

LEFT_FORCE_NAMES = (
    "left_wrench.force.x",
    "left_wrench.force.y",
    "left_wrench.force.z",
)
RIGHT_FORCE_NAMES = (
    "right_wrench.force.x",
    "right_wrench.force.y",
    "right_wrench.force.z",
)
FORCE_NAMES = LEFT_FORCE_NAMES + RIGHT_FORCE_NAMES


def load_dataset(repo_id: str, root: str | None) -> LeRobotDataset:
    print(f"正在加载数据集: {repo_id}")
    if root:
        print(f"  root: {root}")
    dataset = LeRobotDataset(repo_id, root=root)
    print(f"✅ 数据集加载成功  路径: {dataset.root}")
    print(f"  总帧数: {len(dataset)}")
    return dataset


def parse_episode_indices(dataset: LeRobotDataset) -> list[tuple[int, int, int]]:
    """解析 episode_data_index，返回 [(ep_id, start, end), ...]。"""
    ep_idx = dataset.episode_data_index
    if not (isinstance(ep_idx, dict) and "from" in ep_idx and "to" in ep_idx):
        raise ValueError(f"不支持的 episode_data_index 格式: {type(ep_idx)}")

    f = ep_idx["from"]
    t = ep_idx["to"]
    if isinstance(f, torch.Tensor):
        f = f.cpu().numpy().flatten()
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy().flatten()

    return [(i, int(f[i]), int(t[i])) for i in range(min(len(f), len(t)))]


def load_feature_names(dataset: LeRobotDataset, column: str) -> list[str]:
    """从 meta/info.json 读取指定列的维度名称。"""
    info_path = Path(dataset.root) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"找不到数据集元信息文件: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    if column not in features:
        raise ValueError(f"meta/info.json 中没有 {column!r} 字段")

    names = features[column].get("names")
    if not isinstance(names, list):
        raise ValueError(f"meta/info.json 中 {column!r} 没有可用的 names 列表")
    return names


def resolve_force_indices(dataset: LeRobotDataset, column: str) -> tuple[list[int], list[int], list[int]]:
    """按字段名解析左右 force xyz 的索引。"""
    names = load_feature_names(dataset, column)
    name_to_index = {name: i for i, name in enumerate(names)}

    missing = [name for name in FORCE_NAMES if name not in name_to_index]
    if missing:
        raise ValueError(f"{column!r} 缺少 force 字段: {missing}")

    left_indices = [name_to_index[name] for name in LEFT_FORCE_NAMES]
    right_indices = [name_to_index[name] for name in RIGHT_FORCE_NAMES]
    force_indices = left_indices + right_indices

    print(f"  {column} force 索引:")
    print(f"    左臂: {left_indices}")
    print(f"    右臂: {right_indices}")
    return left_indices, right_indices, force_indices


def read_vector_column(hf_ds, column: str) -> np.ndarray:
    """读取 HuggingFace Dataset 的向量列为 float32 numpy 数组。"""
    raw_values = hf_ds[column]
    if isinstance(raw_values[0], torch.Tensor):
        values = np.array([v.cpu().numpy() for v in raw_values], dtype=np.float32)
    else:
        values = np.array(raw_values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{column!r} 期望二维数组，实际 shape={values.shape}")
    return values


def print_force_distribution(
    title: str,
    values: np.ndarray,
    left_indices: list[int],
    force_indices: list[int],
) -> None:
    force_values = values[:, force_indices]
    print(f"\n{title}:")
    for i, idx in enumerate(force_indices):
        arm = "左臂" if idx in left_indices else "右臂"
        axis = ["x", "y", "z"][i % 3]
        print(
            f"  {arm} force.{axis} (dim {idx}): "
            f"min={force_values[:, i].min():.4f}, max={force_values[:, i].max():.4f}"
        )


def compute_force_deltas(dataset: LeRobotDataset, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    """
    遍历所有 episode，计算 force 维度的 episode-start delta。
    每个目标列都减去同一 episode 首帧 observation.state force offset。
    返回每个目标列的完整数组，其中 force 维度已替换为 delta 值。
    """
    hf_ds = dataset.hf_dataset
    if hf_ds is None:
        raise ValueError("数据集没有 hf_dataset")

    missing_columns = [column for column in columns if column not in hf_ds.column_names]
    if missing_columns:
        raise ValueError(f"数据集中缺少列: {missing_columns}")

    episodes = parse_episode_indices(dataset)
    print(f"  共 {len(episodes)} 个 episode，开始计算 force delta...")

    _, _, state_force_indices = resolve_force_indices(dataset, STATE_COLUMN)
    state_values_for_baseline = read_vector_column(hf_ds, STATE_COLUMN)
    force_baselines = {
        ep_id: state_values_for_baseline[start, state_force_indices].copy()
        for ep_id, start, end in episodes
        if start < end
    }

    converted_columns: dict[str, np.ndarray] = {}
    for column in columns:
        print(f"\n正在读取 {column} 数据...")
        left_indices, _, force_indices = resolve_force_indices(dataset, column)
        values = read_vector_column(hf_ds, column)
        print(f"  {column} shape: {values.shape}")
        if max(force_indices) >= values.shape[1]:
            raise ValueError(f"{column!r} force 索引超出维度: shape={values.shape}, indices={force_indices}")

        print_force_distribution(f"{column} 转换前 force 分布", values, left_indices, force_indices)

        for ep_id, start, end in tqdm(episodes, desc=f"处理 {column} episode"):
            if start >= end:
                continue
            values[start:end, force_indices] -= force_baselines[ep_id]

        print_force_distribution(f"{column} 转换后 force delta 分布", values, left_indices, force_indices)
        converted_columns[column] = values

    return converted_columns


def save_dataset(
    dataset: LeRobotDataset,
    converted_columns: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """将修改后的向量列保存回 parquet 文件。"""
    hf_ds = dataset.hf_dataset

    print("\n正在准备修改后的数据...")
    updated_hf_ds = hf_ds
    for column, values in converted_columns.items():
        # 将 numpy array 转为 list of list（HuggingFace Dataset 需要）
        updated_hf_ds = updated_hf_ds.remove_columns([column])
        updated_hf_ds = updated_hf_ds.add_column(column, values.tolist())

    # 复制原数据集目录到输出目录（保留视频、meta 等文件）
    src_root = Path(dataset.root)
    if output_dir != src_root:
        print(f"复制数据集目录结构: {src_root} -> {output_dir}")
        if output_dir.exists():
            print(f"  输出目录已存在，将覆盖: {output_dir}")
            shutil.rmtree(output_dir)
        shutil.copytree(src_root, output_dir)

    # 找到 parquet 文件目录
    data_dir = output_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"找不到 data 目录: {data_dir}")

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"data 目录下没有 parquet 文件: {data_dir}")

    print(f"找到 {len(parquet_files)} 个 parquet 文件，将重新写入...")

    # LeRobot v2 将数据分片存储为多个 parquet，逐一对应写回
    # 先按文件内行号分片
    if len(parquet_files) == 1:
        # 单文件直接保存
        updated_hf_ds.to_parquet(str(parquet_files[0]))
        print(f"✅ 已保存: {parquet_files[0]}")
    else:
        # 多文件：按原始分片大小依次切片保存
        import pyarrow.parquet as pq

        offset = 0
        for pq_file in tqdm(parquet_files, desc="写入 parquet"):
            original_table = pq.read_table(str(pq_file))
            chunk_size = len(original_table)
            chunk_ds = updated_hf_ds.select(range(offset, offset + chunk_size))
            chunk_ds.to_parquet(str(pq_file))
            offset += chunk_size

        print(f"✅ 已写入 {len(parquet_files)} 个 parquet 文件，共 {offset} 帧")


def validate_conversion(dataset_out: LeRobotDataset, columns: tuple[str, ...]) -> None:
    """简单验证转换结果。

    observation.state 使用自身首帧作为 offset，因此每个 episode 首帧 force delta 应为 0。
    action 使用同一个 state offset，但首帧 action force 可能对应未来目标，不要求为 0。
    """
    print("\n正在验证转换结果...")
    hf_ds = dataset_out.hf_dataset
    episodes = parse_episode_indices(dataset_out)

    errors = 0
    for column in columns:
        _, _, force_indices = resolve_force_indices(dataset_out, column)
        for ep_id, start, _ in episodes[:min(10, len(episodes))]:
            values = hf_ds[start][column]
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            else:
                values = np.array(values)
            force_delta_first = values[force_indices]
            if not np.all(np.isfinite(force_delta_first)):
                print(f"  ⚠️ {column} episode {ep_id} 首帧 force delta 存在非有限值: {force_delta_first}")
                errors += 1
                continue
            if column == STATE_COLUMN and not np.allclose(force_delta_first, 0.0, atol=1e-5):
                print(f"  ⚠️ {column} episode {ep_id} 第一帧 force delta 不为 0: {force_delta_first}")
                errors += 1

    if errors == 0:
        print("✅ 验证通过：state 首帧为 0，目标列 force delta 均为有限值")
    else:
        print(f"❌ 验证发现 {errors} 个 episode 异常")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Darwin02 force 数据转换为 episode-start delta 模式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_id", type=str, required=True,
                        help="LeRobot 数据集 repo id")
    parser.add_argument("--root", type=str, default=None,
                        help="数据集根目录（本地路径）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录。不指定则 in-place 覆盖原数据集")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅预览分布变化，不写入文件")
    parser.add_argument("--columns", nargs="+", choices=DEFAULT_COLUMNS, default=list(DEFAULT_COLUMNS),
                        help="需要转换的向量列")
    args = parser.parse_args()

    dataset = load_dataset(args.repo_id, args.root)
    columns = tuple(args.columns)

    converted_columns = compute_force_deltas(dataset, columns)

    if args.dry_run:
        print("\n[dry_run] 预览完成，未写入任何文件")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(dataset.root)

    save_dataset(dataset, converted_columns, output_dir)

    # 重新加载验证
    print("\n重新加载数据集进行验证...")
    dataset_out = LeRobotDataset(args.repo_id, root=str(output_dir.parent)
                                 if output_dir.name == args.repo_id.split("/")[-1]
                                 else str(output_dir))
    validate_conversion(dataset_out, columns)

    print(f"\n完成！转换后的数据集位于: {output_dir}")
    print("请重新运行 compute_norm_stats.py 更新归一化统计文件。")


if __name__ == "__main__":
    main()
