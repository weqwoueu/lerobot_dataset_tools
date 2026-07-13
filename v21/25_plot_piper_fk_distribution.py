#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将双 Piper 数据集的关节 state 转为末端位姿，并分析空间停留分布。

默认使用 AgileX 官方 piper_sdk 中 ``C_PiperForwardKinematics`` 的 DH 参数，
其中关节 2、3 启用 2 度补偿（``dh_is_offset=1``）。程序会读取每一个
``observation.state``，对左右臂分别执行 FK，并输出：

1. 全量末端位姿 ``end_effector_poses.npz``；
2. 可旋转缩放的三维图 ``end_effector_3d_interactive.html``；
3. 三维停留密度图 ``end_effector_3d_distribution.png/.pdf``；
4. XY/XZ/YZ 密度投影 ``end_effector_density_projections.png/.pdf``；
5. 高频停留体素 ``top_occupied_voxels.csv`` 和汇总 ``summary.json``。

默认 FK 末端是官方定义的 joint6/link6 原点。如果要分析夹爪前端，可通过
``--tcp-offset 0 0 0.13503`` 在末端局部坐标系中增加 TCP 偏移。

示例：
    .venv/bin/python 25_plot_piper_fk_distribution.py

    .venv/bin/python 25_plot_piper_fk_distribution.py \
        --voxel-size 0.01 \
        --tcp-offset 0 0 0.13503

如果已知双臂基座在同一世界坐标系中的外参，可分别设置：
    --left-base-pose X Y Z ROLL_DEG PITCH_DEG YAW_DEG
    --right-base-pose X Y Z ROLL_DEG PITCH_DEG YAW_DEG

当前双臂安装默认以左臂基座为分析坐标系原点，右臂基座位于左臂 Y 轴负方向
0.71 m，且两臂基座坐标轴方向平行。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import LogNorm
from plotly.subplots import make_subplots

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("缺少 pyarrow，请在项目的 LeRobot Python 环境中运行该脚本。") from exc

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable


DEFAULT_DATASET_DIR = Path(
    "/home/standard/agilex/lerobot/piperx/dagger/"
    "piperx_grab_bigbox_yellow_0529_0703_nonidle"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "piper_fk_distribution"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_LEFT_BASE_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_RIGHT_BASE_POSE = (0.0, -0.71, 0.0, 0.0, 0.0, 0.0)

LEFT_JOINT_NAMES = [f"left_joint_{index}_pos" for index in range(1, 7)]
RIGHT_JOINT_NAMES = [f"right_joint_{index}_pos" for index in range(1, 7)]
POSE_COLUMNS = ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"]

# AgileX piper_sdk.C_PiperForwardKinematics 的官方 DH 参数，长度单位转换为米。
DH_A_M = np.asarray([0.0, 0.0, 0.28503, -0.02198, 0.0, 0.0], dtype=np.float64)
DH_ALPHA_RAD = np.asarray(
    [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2],
    dtype=np.float64,
)
DH_D_M = np.asarray([0.123, 0.0, 0.0, 0.25075, 0.0, 0.091], dtype=np.float64)
DH_THETA_OFFSET_RAD = np.asarray(
    [0.0, -math.radians(172.22), -math.radians(102.78), 0.0, 0.0, 0.0],
    dtype=np.float64,
)
DH_THETA_NO_OFFSET_RAD = np.asarray(
    [0.0, -math.radians(174.22), -math.radians(100.78), 0.0, 0.0, 0.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class DatasetStates:
    left_joints: np.ndarray
    right_joints: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray
    timestamp_s: np.ndarray
    parquet_files: int


@dataclass(frozen=True)
class VoxelDistribution:
    centers_m: np.ndarray
    counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取双 Piper LeRobot 数据集，执行 FK 并绘制末端空间停留分布。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"LeRobot 数据集目录，默认：{DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"分析结果目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--state-key",
        default=DEFAULT_STATE_KEY,
        help=f"关节状态字段，默认：{DEFAULT_STATE_KEY}",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="三维停留密度体素边长，单位米，默认 0.01（1 cm）。",
    )
    parser.add_argument(
        "--tcp-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="末端局部坐标系中的 TCP 偏移，单位米，默认 0 0 0（joint6 原点）。",
    )
    parser.add_argument(
        "--left-base-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=DEFAULT_LEFT_BASE_POSE,
        help="左臂基座在分析坐标系中的位姿：米、度。默认单位阵。",
    )
    parser.add_argument(
        "--right-base-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=DEFAULT_RIGHT_BASE_POSE,
        help="右臂基座在分析坐标系中的位姿：米、度。默认 0 -0.71 0 0 0 0。",
    )
    parser.add_argument(
        "--no-dh-offset",
        action="store_true",
        help="关闭 Piper 关节 2、3 的官方 2 度 DH 补偿。默认启用补偿。",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="只处理前 N 个 episode，主要用于快速验证；默认处理全部。",
    )
    parser.add_argument(
        "--max-plot-voxels",
        type=int,
        default=100_000,
        help="每条机械臂在三维密度图中最多显示的体素数，默认 100000。",
    )
    parser.add_argument(
        "--max-overlay-points",
        type=int,
        default=150_000,
        help="左右臂叠加图中每条臂最多绘制的状态点数，默认 150000。",
    )
    parser.add_argument(
        "--top-voxels",
        type=int,
        default=200,
        help="CSV 中每条臂保存的高频体素数，默认 200。",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="用于将帧数换算为停留秒数；默认读取 meta/info.json。",
    )
    parser.add_argument(
        "--no-save-poses",
        action="store_true",
        help="不保存全量 end_effector_poses.npz，仅生成统计和图片。",
    )
    parser.add_argument(
        "--no-interactive-html",
        action="store_true",
        help="不生成可旋转、缩放的交互式三维 HTML。默认生成自包含 HTML。",
    )
    args = parser.parse_args()

    if args.voxel_size <= 0:
        parser.error("--voxel-size 必须大于 0")
    for name in ("max_episodes", "max_plot_voxels", "max_overlay_points", "top_voxels"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps 必须大于 0")
    return args


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def feature_names(info: dict[str, Any], state_key: str) -> list[str] | None:
    names = info.get("features", {}).get(state_key, {}).get("names")
    if not names:
        return None
    if len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    return [str(name) for name in names]


def resolve_joint_indices(info: dict[str, Any], state_key: str) -> tuple[list[int], list[int]]:
    names = feature_names(info, state_key)
    if names is None:
        print("警告：info.json 中没有 state 维度名称，按 14 维双 Piper 标准顺序读取。")
        return list(range(6)), list(range(7, 13))

    missing = [name for name in LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES if name not in names]
    if missing:
        raise ValueError(f"{state_key} 缺少所需关节维度：{missing}")
    return [names.index(name) for name in LEFT_JOINT_NAMES], [names.index(name) for name in RIGHT_JOINT_NAMES]


def fixed_list_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        values = array.values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float64).reshape(len(array), array.type.list_size)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        return np.asarray(array.to_pylist(), dtype=np.float64)
    values = np.asarray(array.to_numpy(zero_copy_only=False))
    if values.ndim == 1:
        values = np.stack(values)
    return values.astype(np.float64, copy=False)


def scalar_column(table: pa.Table, name: str, default: np.ndarray) -> np.ndarray:
    if name not in table.column_names:
        return default
    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))


def episode_index_from_path(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def find_parquet_files(dataset_dir: Path, max_episodes: int | None) -> list[Path]:
    files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有在数据集 data 目录中找到 parquet：{dataset_dir / 'data'}")
    if max_episodes is not None:
        files = files[:max_episodes]
    return files


def load_dataset_states(
    dataset_dir: Path,
    state_key: str,
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    max_episodes: int | None,
) -> DatasetStates:
    parquet_files = find_parquet_files(dataset_dir, max_episodes)
    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    episode_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []
    timestamp_chunks: list[np.ndarray] = []

    for path in tqdm(parquet_files, desc="读取 state", unit="episode"):
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        if state_key not in schema_names:
            raise KeyError(f"{path} 中缺少字段 {state_key}")
        columns = [state_key]
        columns.extend(name for name in ("episode_index", "frame_index", "timestamp") if name in schema_names)
        table = pq.read_table(path, columns=columns)
        states = fixed_list_to_numpy(table[state_key])
        required_dimension = max(max(left_indices), max(right_indices)) + 1
        if states.ndim != 2 or states.shape[1] < required_dimension:
            raise ValueError(
                f"{path} 的 {state_key} shape={states.shape}，无法读取所需的 {required_dimension} 个维度"
            )

        rows = states.shape[0]
        episode_default = np.full(rows, episode_index_from_path(path), dtype=np.int64)
        frame_default = np.arange(rows, dtype=np.int64)
        timestamp_default = np.full(rows, np.nan, dtype=np.float64)
        left_chunks.append(states[:, left_indices])
        right_chunks.append(states[:, right_indices])
        episode_chunks.append(scalar_column(table, "episode_index", episode_default).astype(np.int64))
        frame_chunks.append(scalar_column(table, "frame_index", frame_default).astype(np.int64))
        timestamp_chunks.append(scalar_column(table, "timestamp", timestamp_default).astype(np.float64))

    return DatasetStates(
        left_joints=np.concatenate(left_chunks, axis=0),
        right_joints=np.concatenate(right_chunks, axis=0),
        episode_index=np.concatenate(episode_chunks),
        frame_index=np.concatenate(frame_chunks),
        timestamp_s=np.concatenate(timestamp_chunks),
        parquet_files=len(parquet_files),
    )


def dh_transform_batch(alpha: float, a: float, theta: np.ndarray, d: float) -> np.ndarray:
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    transform = np.zeros((len(theta), 4, 4), dtype=np.float64)
    transform[:, 0, 0] = cos_theta
    transform[:, 0, 1] = -sin_theta
    transform[:, 0, 3] = a
    transform[:, 1, 0] = sin_theta * cos_alpha
    transform[:, 1, 1] = cos_theta * cos_alpha
    transform[:, 1, 2] = -sin_alpha
    transform[:, 1, 3] = -sin_alpha * d
    transform[:, 2, 0] = sin_theta * sin_alpha
    transform[:, 2, 1] = cos_theta * sin_alpha
    transform[:, 2, 2] = cos_alpha
    transform[:, 2, 3] = cos_alpha * d
    transform[:, 3, 3] = 1.0
    return transform


def rpy_transform(pose_xyz_rpy_deg: Sequence[float]) -> np.ndarray:
    x, y, z, roll_deg, pitch_deg, yaw_deg = pose_xyz_rpy_deg
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [x, y, z]
    return transform


def matrix_batch_to_pose(transforms: np.ndarray) -> np.ndarray:
    rotation = transforms[:, :3, :3]
    pose = np.empty((len(transforms), 6), dtype=np.float64)
    pose[:, :3] = transforms[:, :3, 3]

    r20 = rotation[:, 2, 0]
    positive_singularity = r20 < -1.0 + 1e-4
    negative_singularity = r20 > 1.0 - 1e-4
    regular = ~(positive_singularity | negative_singularity)

    pose[regular, 4] = np.arctan2(
        -r20[regular],
        np.hypot(rotation[regular, 0, 0], rotation[regular, 1, 0]),
    )
    pose[regular, 5] = np.arctan2(rotation[regular, 1, 0], rotation[regular, 0, 0])
    pose[regular, 3] = np.arctan2(rotation[regular, 2, 1], rotation[regular, 2, 2])

    pose[positive_singularity, 4] = math.pi / 2
    pose[positive_singularity, 5] = 0.0
    pose[positive_singularity, 3] = np.arctan2(
        rotation[positive_singularity, 0, 1],
        rotation[positive_singularity, 1, 1],
    )

    pose[negative_singularity, 4] = -math.pi / 2
    pose[negative_singularity, 5] = 0.0
    pose[negative_singularity, 3] = -np.arctan2(
        rotation[negative_singularity, 0, 1],
        rotation[negative_singularity, 1, 1],
    )
    return pose


def piper_fk_batch(
    joints_rad: np.ndarray,
    *,
    dh_offset: bool = True,
    tcp_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
    base_pose_xyz_rpy_deg: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """批量计算 Piper 末端位姿，返回 [x,y,z,roll,pitch,yaw]（米、弧度）。"""
    joints = np.asarray(joints_rad, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 6:
        raise ValueError(f"Piper FK 输入应为 (N, 6)，实际为 {joints.shape}")
    theta_offset = DH_THETA_OFFSET_RAD if dh_offset else DH_THETA_NO_OFFSET_RAD
    transforms = np.broadcast_to(np.eye(4), (len(joints), 4, 4)).copy()
    for joint_index in range(6):
        link_transform = dh_transform_batch(
            DH_ALPHA_RAD[joint_index],
            DH_A_M[joint_index],
            joints[:, joint_index] + theta_offset[joint_index],
            DH_D_M[joint_index],
        )
        transforms = transforms @ link_transform

    tcp_transform = np.eye(4, dtype=np.float64)
    tcp_transform[:3, 3] = np.asarray(tcp_offset_m, dtype=np.float64)
    base_transform = rpy_transform(base_pose_xyz_rpy_deg)
    transforms = base_transform[None, :, :] @ transforms @ tcp_transform[None, :, :]
    return matrix_batch_to_pose(transforms)


def valid_positions(pose: np.ndarray) -> np.ndarray:
    mask = np.isfinite(pose[:, :3]).all(axis=1)
    return pose[mask, :3]


def voxelize(points_m: np.ndarray, voxel_size_m: float) -> VoxelDistribution:
    if len(points_m) == 0:
        raise ValueError("没有有限的末端位置可用于体素统计")
    voxel_indices = np.floor(points_m / voxel_size_m).astype(np.int64)
    unique_indices, counts = np.unique(voxel_indices, axis=0, return_counts=True)
    centers = (unique_indices.astype(np.float64) + 0.5) * voxel_size_m
    return VoxelDistribution(centers_m=centers, counts=counts.astype(np.int64))


def select_plot_voxels(distribution: VoxelDistribution, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    if len(distribution.counts) <= maximum:
        return distribution.centers_m, distribution.counts
    selected = np.argpartition(distribution.counts, -maximum)[-maximum:]
    return distribution.centers_m[selected], distribution.counts[selected]


def common_limits(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([left, right], axis=0)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    span = np.maximum(maximum - minimum, 0.05)
    padding = span * 0.06
    return minimum - padding, maximum + padding


def format_3d_axis(axis: Any, title: str, limits: tuple[np.ndarray, np.ndarray]) -> None:
    minimum, maximum = limits
    axis.set_title(title)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_xlim(minimum[0], maximum[0])
    axis.set_ylim(minimum[1], maximum[1])
    axis.set_zlim(minimum[2], maximum[2])
    axis.set_box_aspect(np.maximum(maximum - minimum, 0.05))
    axis.view_init(elev=24, azim=-58)


def plot_arm_voxels(
    figure: plt.Figure,
    axis: Any,
    distribution: VoxelDistribution,
    fps: float,
    maximum: int,
    title: str,
    cmap: str,
    limits: tuple[np.ndarray, np.ndarray],
) -> None:
    centers, counts = select_plot_voxels(distribution, maximum)
    dwell_seconds = counts / fps
    scale = np.log1p(counts) / max(float(np.log1p(np.max(counts))), 1.0)
    scatter = axis.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=dwell_seconds,
        cmap=cmap,
        norm=LogNorm(vmin=max(1.0 / fps, float(np.min(dwell_seconds))), vmax=float(np.max(dwell_seconds))),
        s=3.0 + 24.0 * scale,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    format_3d_axis(axis, f"{title}\noccupied voxels: {len(distribution.counts):,}", limits)
    colorbar = figure.colorbar(scatter, ax=axis, shrink=0.72, pad=0.08)
    colorbar.set_label("Accumulated dwell time per voxel (s)")


def sample_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def save_figure(figure: plt.Figure, output_paths: Sequence[Path]) -> None:
    for output_path in output_paths:
        figure.savefig(output_path, dpi=220)


def save_3d_distribution_plot(
    output_paths: Sequence[Path],
    left_points: np.ndarray,
    right_points: np.ndarray,
    left_voxels: VoxelDistribution,
    right_voxels: VoxelDistribution,
    fps: float,
    max_plot_voxels: int,
    max_overlay_points: int,
) -> None:
    limits = common_limits(left_points, right_points)
    figure = plt.figure(figsize=(20, 7), layout="constrained")
    left_axis = figure.add_subplot(1, 3, 1, projection="3d")
    right_axis = figure.add_subplot(1, 3, 2, projection="3d")
    overlay_axis = figure.add_subplot(1, 3, 3, projection="3d")

    plot_arm_voxels(
        figure,
        left_axis,
        left_voxels,
        fps,
        max_plot_voxels,
        "Left arm dwell density",
        "viridis",
        limits,
    )
    plot_arm_voxels(
        figure,
        right_axis,
        right_voxels,
        fps,
        max_plot_voxels,
        "Right arm dwell density",
        "plasma",
        limits,
    )

    left_sample = sample_points(left_points, max_overlay_points)
    right_sample = sample_points(right_points, max_overlay_points)
    overlay_axis.scatter(
        left_sample[:, 0],
        left_sample[:, 1],
        left_sample[:, 2],
        s=0.5,
        alpha=0.10,
        color="#0072B2",
        label=f"Left ({len(left_sample):,} shown)",
        rasterized=True,
    )
    overlay_axis.scatter(
        right_sample[:, 0],
        right_sample[:, 1],
        right_sample[:, 2],
        s=0.5,
        alpha=0.10,
        color="#D55E00",
        label=f"Right ({len(right_sample):,} shown)",
        rasterized=True,
    )
    format_3d_axis(overlay_axis, "Left/right workspace overlay", limits)
    overlay_axis.legend(loc="upper right", markerscale=8)
    figure.suptitle(
        f"Piper end-effector spatial distribution (all {len(left_points):,} valid states used for density)",
        fontsize=14,
    )
    save_figure(figure, output_paths)
    plt.close(figure)


def plotly_marker_sizes(counts: np.ndarray) -> np.ndarray:
    denominator = max(float(np.log1p(np.max(counts))), 1.0)
    return 2.0 + 6.0 * np.log1p(counts) / denominator


def interactive_voxel_trace(
    distribution: VoxelDistribution,
    fps: float,
    maximum: int,
    *,
    name: str,
    colorscale: str,
    colorbar_x: float | None,
    scene: str,
    fixed_color: str | None = None,
) -> go.Scatter3d:
    centers, counts = select_plot_voxels(distribution, maximum)
    dwell_seconds = counts / fps
    customdata = np.column_stack([counts, dwell_seconds])
    marker: dict[str, Any] = {
        "size": plotly_marker_sizes(counts),
        "opacity": 0.72,
        "line": {"width": 0},
    }
    if fixed_color is None:
        marker.update(
            {
                "color": np.log10(dwell_seconds),
                "colorscale": colorscale,
                "showscale": True,
                "colorbar": {
                    "title": {"text": "log10 dwell (s)"},
                    "x": colorbar_x,
                    "len": 0.72,
                    "thickness": 14,
                },
            }
        )
    else:
        marker.update({"color": fixed_color, "size": 2.5, "opacity": 0.45})

    return go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode="markers",
        marker=marker,
        customdata=customdata,
        name=name,
        scene=scene,
        hovertemplate=(
            f"{name}<br>"
            "X: %{x:.4f} m<br>"
            "Y: %{y:.4f} m<br>"
            "Z: %{z:.4f} m<br>"
            "Frames: %{customdata[0]:,.0f}<br>"
            "Dwell: %{customdata[1]:.3f} s<extra></extra>"
        ),
    )


def save_interactive_3d_plot(
    output_path: Path,
    left_voxels: VoxelDistribution,
    right_voxels: VoxelDistribution,
    fps: float,
    max_plot_voxels: int,
    total_states: int,
) -> None:
    figure = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Left arm dwell density", "Right arm dwell density", "Workspace overlay"),
        horizontal_spacing=0.035,
    )
    figure.add_trace(
        interactive_voxel_trace(
            left_voxels,
            fps,
            max_plot_voxels,
            name="Left arm",
            colorscale="Viridis",
            colorbar_x=0.305,
            scene="scene",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        interactive_voxel_trace(
            right_voxels,
            fps,
            max_plot_voxels,
            name="Right arm",
            colorscale="Plasma",
            colorbar_x=0.65,
            scene="scene2",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        interactive_voxel_trace(
            left_voxels,
            fps,
            max_plot_voxels,
            name="Left arm",
            colorscale="Viridis",
            colorbar_x=None,
            scene="scene3",
            fixed_color="#0072B2",
        ),
        row=1,
        col=3,
    )
    figure.add_trace(
        interactive_voxel_trace(
            right_voxels,
            fps,
            max_plot_voxels,
            name="Right arm",
            colorscale="Plasma",
            colorbar_x=None,
            scene="scene3",
            fixed_color="#D55E00",
        ),
        row=1,
        col=3,
    )

    camera = {"eye": {"x": 1.45, "y": -1.45, "z": 1.05}}
    axis_style = {
        "xaxis": {"title": "X (m)", "showspikes": False},
        "yaxis": {"title": "Y (m)", "showspikes": False},
        "zaxis": {"title": "Z (m)", "showspikes": False},
        "aspectmode": "data",
        "camera": camera,
    }
    figure.update_layout(
        title={
            "text": f"Piper end-effector spatial distribution ({total_states:,} valid states)",
            "x": 0.5,
        },
        scene=axis_style,
        scene2=axis_style,
        scene3=axis_style,
        template="plotly_white",
        height=760,
        margin={"l": 10, "r": 10, "t": 85, "b": 10},
        legend={"orientation": "h", "x": 0.83, "xanchor": "center", "y": 0.02},
        uirevision="piper-fk-distribution",
    )
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )


def projection_edges(left: np.ndarray, right: np.ndarray, voxel_size: float) -> list[np.ndarray]:
    points = np.concatenate([left, right], axis=0)
    edges = []
    for dimension in range(3):
        lower = math.floor(float(np.min(points[:, dimension])) / voxel_size) * voxel_size
        upper = math.ceil(float(np.max(points[:, dimension])) / voxel_size) * voxel_size
        bins = max(1, int(round((upper - lower) / voxel_size)))
        bins = min(bins, 250)
        if upper <= lower:
            upper = lower + voxel_size
        edges.append(np.linspace(lower, upper, bins + 1))
    return edges


def save_projection_plot(
    output_paths: Sequence[Path],
    left_points: np.ndarray,
    right_points: np.ndarray,
    voxel_size: float,
) -> None:
    edges = projection_edges(left_points, right_points, voxel_size)
    projections = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
    arms = [(left_points, "Left arm", "viridis"), (right_points, "Right arm", "plasma")]
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")

    for row, (points, arm_name, cmap) in enumerate(arms):
        for column, (x_index, y_index, plane) in enumerate(projections):
            axis = axes[row, column]
            histogram = axis.hist2d(
                points[:, x_index],
                points[:, y_index],
                bins=[edges[x_index], edges[y_index]],
                cmap=cmap,
                norm=LogNorm(vmin=1),
                rasterized=True,
            )
            axis.set_title(f"{arm_name}: {plane} dwell density")
            axis.set_xlabel(f"{'XYZ'[x_index]} (m)")
            axis.set_ylabel(f"{'XYZ'[y_index]} (m)")
            axis.set_aspect("equal", adjustable="box")
            colorbar = figure.colorbar(histogram[3], ax=axis)
            colorbar.set_label("Frame count per 2D bin")

    figure.suptitle("Piper end-effector density projections (all valid states)", fontsize=14)
    save_figure(figure, output_paths)
    plt.close(figure)


def position_summary(points: np.ndarray) -> dict[str, Any]:
    quantiles = np.quantile(points, [0.01, 0.05, 0.5, 0.95, 0.99], axis=0)
    return {
        "valid_frames": int(len(points)),
        "min_m": np.min(points, axis=0).tolist(),
        "max_m": np.max(points, axis=0).tolist(),
        "mean_m": np.mean(points, axis=0).tolist(),
        "std_m": np.std(points, axis=0).tolist(),
        "quantiles_m": {
            label: value.tolist()
            for label, value in zip(("p01", "p05", "p50", "p95", "p99"), quantiles, strict=True)
        },
    }


def voxel_summary(distribution: VoxelDistribution, fps: float) -> dict[str, Any]:
    busiest = int(np.argmax(distribution.counts))
    return {
        "occupied_voxels": int(len(distribution.counts)),
        "busiest_voxel_center_m": distribution.centers_m[busiest].tolist(),
        "busiest_voxel_frames": int(distribution.counts[busiest]),
        "busiest_voxel_dwell_seconds": float(distribution.counts[busiest] / fps),
    }


def save_top_voxels(
    output_path: Path,
    distributions: Sequence[tuple[str, VoxelDistribution]],
    fps: float,
    maximum: int,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "arm",
                "rank",
                "x_center_m",
                "y_center_m",
                "z_center_m",
                "frame_count",
                "dwell_seconds",
                "frame_percentage",
            ]
        )
        for arm, distribution in distributions:
            order = np.argsort(distribution.counts)[::-1][:maximum]
            total_frames = int(np.sum(distribution.counts))
            for rank, index in enumerate(order, start=1):
                center = distribution.centers_m[index]
                count = int(distribution.counts[index])
                writer.writerow(
                    [
                        arm,
                        rank,
                        *[f"{value:.6f}" for value in center],
                        count,
                        f"{count / fps:.6f}",
                        f"{100.0 * count / total_frames:.8f}",
                    ]
                )


def save_poses(
    output_path: Path,
    states: DatasetStates,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
) -> None:
    np.savez_compressed(
        output_path,
        pose_columns=np.asarray(POSE_COLUMNS),
        episode_index=states.episode_index,
        frame_index=states.frame_index,
        timestamp_s=states.timestamp_s.astype(np.float32),
        left_pose=left_pose.astype(np.float32),
        right_pose=right_pose.astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"数据集缺少 meta/info.json：{info_path}")

    info = read_json(info_path)
    left_indices, right_indices = resolve_joint_indices(info, args.state_key)
    fps = float(args.fps if args.fps is not None else info.get("fps", 30.0))
    if fps <= 0:
        raise ValueError(f"fps 必须大于 0，实际为 {fps}")

    print(f"数据集：{dataset_dir}")
    print(f"state 字段：{args.state_key}")
    print(f"左臂关节索引：{left_indices}；右臂关节索引：{right_indices}")
    states = load_dataset_states(
        dataset_dir,
        args.state_key,
        left_indices,
        right_indices,
        args.max_episodes,
    )
    print(f"已读取 {states.parquet_files} 个 episode，共 {len(states.left_joints):,} 个 state")

    dh_offset = not args.no_dh_offset
    print(f"执行向量化 FK（DH 2 度补偿：{'启用' if dh_offset else '关闭'}）...")
    left_pose = piper_fk_batch(
        states.left_joints,
        dh_offset=dh_offset,
        tcp_offset_m=args.tcp_offset,
        base_pose_xyz_rpy_deg=args.left_base_pose,
    )
    right_pose = piper_fk_batch(
        states.right_joints,
        dh_offset=dh_offset,
        tcp_offset_m=args.tcp_offset,
        base_pose_xyz_rpy_deg=args.right_base_pose,
    )
    left_points = valid_positions(left_pose)
    right_points = valid_positions(right_pose)
    if len(left_points) == 0 or len(right_points) == 0:
        raise ValueError("FK 结果中没有可用的有限位置，请检查 state 内容和关节索引")

    left_voxels = voxelize(left_points, args.voxel_size)
    right_voxels = voxelize(right_points, args.voxel_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    poses_path = output_dir / "end_effector_poses.npz"
    interactive_path = output_dir / "end_effector_3d_interactive.html"
    plot_3d_png_path = output_dir / "end_effector_3d_distribution.png"
    plot_3d_pdf_path = output_dir / "end_effector_3d_distribution.pdf"
    projection_png_path = output_dir / "end_effector_density_projections.png"
    projection_pdf_path = output_dir / "end_effector_density_projections.pdf"
    voxels_path = output_dir / "top_occupied_voxels.csv"
    summary_path = output_dir / "summary.json"

    if not args.no_save_poses:
        print(f"保存全量末端位姿：{poses_path}")
        save_poses(poses_path, states, left_pose, right_pose)

    print("绘制三维停留密度图...")
    save_3d_distribution_plot(
        [plot_3d_png_path, plot_3d_pdf_path],
        left_points,
        right_points,
        left_voxels,
        right_voxels,
        fps,
        args.max_plot_voxels,
        args.max_overlay_points,
    )
    if not args.no_interactive_html:
        print("生成可交互三维图...")
        save_interactive_3d_plot(
            interactive_path,
            left_voxels,
            right_voxels,
            fps,
            args.max_plot_voxels,
            min(len(left_points), len(right_points)),
        )
    print("绘制二维密度投影...")
    save_projection_plot(
        [projection_png_path, projection_pdf_path],
        left_points,
        right_points,
        args.voxel_size,
    )
    save_top_voxels(
        voxels_path,
        [("left", left_voxels), ("right", right_voxels)],
        fps,
        args.top_voxels,
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "state_key": args.state_key,
        "episodes_processed": states.parquet_files,
        "states_processed": int(len(states.left_joints)),
        "fps": fps,
        "voxel_size_m": args.voxel_size,
        "fk_model": {
            "source": "AgileX piper_sdk C_PiperForwardKinematics",
            "dh_offset_enabled": dh_offset,
            "tcp_offset_m": list(args.tcp_offset),
            "pose_units": {"position": "meter", "orientation_rpy": "radian"},
        },
        "base_pose_xyz_m_rpy_deg": {
            "left": list(args.left_base_pose),
            "right": list(args.right_base_pose),
        },
        "left": {
            **position_summary(left_points),
            **voxel_summary(left_voxels, fps),
            "invalid_frames": int(len(left_pose) - len(left_points)),
        },
        "right": {
            **position_summary(right_points),
            **voxel_summary(right_voxels, fps),
            "invalid_frames": int(len(right_pose) - len(right_points)),
        },
        "outputs": {
            "poses_npz": None if args.no_save_poses else str(poses_path),
            "interactive_3d_html": None if args.no_interactive_html else str(interactive_path),
            "distribution_3d_png": str(plot_3d_png_path),
            "distribution_3d_pdf": str(plot_3d_pdf_path),
            "density_projections_png": str(projection_png_path),
            "density_projections_pdf": str(projection_pdf_path),
            "top_voxels_csv": str(voxels_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"完成，结果目录：{output_dir}")
    if not args.no_interactive_html:
        print(f"可交互三维图：{interactive_path}")
    print(f"三维分布图：{plot_3d_png_path} / {plot_3d_pdf_path}")
    print(f"密度投影图：{projection_png_path} / {projection_pdf_path}")
    print(f"高频体素表：{voxels_path}")
    print(f"汇总信息：{summary_path}")


if __name__ == "__main__":
    main()
