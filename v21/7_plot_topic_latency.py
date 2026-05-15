#!/usr/bin/env python3
"""Plot per-topic transport latency (bag_timestamp - header_timestamp) for all topics in a rosbag.

Usage:
    conda activate lerobot21
    source /opt/ros/humble/setup.bash
    cd ~/workspace/darwin02/ros_ws && source install/setup.bash
    python tools/plot_topic_latency.py \
        --input-bag /home/standard/darwin02/darwin02_0401/record_20260401_142124_293022 \
        --output plot_latency.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


TOPIC_SHORT_NAMES = {
    "/head_camera/color/image_raw": "head_cam",
    "/left_camera/camera/color/image_rect_raw": "left_wrist_cam",
    "/right_camera/camera/color/image_rect_raw": "right_wrist_cam",
    "/joint_states": "joint_states",
    "/chassis_status": "chassis",
    "/darwin02_manager_main/left_suction_pressure": "left_suction",
    "/darwin02_manager_main/right_suction_pressure": "right_suction",
    "/darwin02_manager_main/right_gripper/suction_cup_sensor": "right_cup",
    "/darwin02_manager_main/left_gripper/suction_cup_sensor": "left_cup",
    "/left_ft_sensor_broadcaster/wrench": "left_wrench",
    "/right_ft_sensor_broadcaster/wrench": "right_wrench",
}


def _extract_header_ns(msg) -> int | None:
    """Try to get header.stamp from a ROS message; return nanoseconds or None."""
    header = getattr(msg, "header", None)
    if header is None:
        return None
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    ns = int(sec) * 10**9 + int(nanosec)
    if ns == 0:
        return None
    return ns


def collect_latencies(
    bag_path: str, storage_id: str,
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Return {topic: (time_seconds_from_start[], latency_ms[])}."""
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id=storage_id),
        ConverterOptions("", ""),
    )
    topic_meta = {t.name: t.type for t in reader.get_all_topics_and_types()}
    target_topics = {t for t in TOPIC_SHORT_NAMES if t in topic_meta}

    msg_classes = {}
    for topic in target_topics:
        try:
            msg_classes[topic] = get_message(topic_meta[topic])
        except Exception:
            pass

    bag_ts_map: Dict[str, List[int]] = {t: [] for t in target_topics}
    header_ts_map: Dict[str, List[int]] = {t: [] for t in target_topics}

    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic not in msg_classes:
            continue
        msg = deserialize_message(data, msg_classes[topic])
        h_ns = _extract_header_ns(msg)
        if h_ns is None:
            continue
        bag_ts_map[topic].append(int(ts))
        header_ts_map[topic].append(h_ns)

    global_start = min(
        (lst[0] for lst in header_ts_map.values() if lst), default=0,
    )

    result: Dict[str, Tuple[List[float], List[float]]] = {}
    for topic in target_topics:
        if not bag_ts_map[topic]:
            continue
        bt = np.asarray(bag_ts_map[topic], dtype=np.float64)
        ht = np.asarray(header_ts_map[topic], dtype=np.float64)
        time_s = (ht - global_start) / 1e9
        latency_ms = (bt - ht) / 1e6
        result[topic] = (time_s.tolist(), latency_ms.tolist())
    return result


def plot_latencies(
    latencies: Dict[str, Tuple[List[float], List[float]]],
    output_path: str,
    bag_name: str,
) -> None:
    camera_topics = [t for t in latencies if "camera" in t]
    other_topics = [t for t in latencies if "camera" not in t]

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    fig.suptitle(f"Topic Transport Latency  (bag - header)\n{bag_name}", fontsize=14, y=0.98)

    cmap_cam = plt.cm.Set1
    cmap_other = plt.cm.tab10

    ax_cam = axes[0]
    ax_cam.set_title("Camera Topics", fontsize=12, fontweight="bold")
    for i, topic in enumerate(sorted(camera_topics)):
        t, lat = latencies[topic]
        label = TOPIC_SHORT_NAMES.get(topic, topic.split("/")[-1])
        color = cmap_cam(i / max(len(camera_topics), 1))
        ax_cam.plot(t, lat, linewidth=0.6, alpha=0.8, label=label, color=color)
        median_lat = float(np.median(lat))
        ax_cam.axhline(median_lat, linestyle="--", linewidth=0.8, alpha=0.5, color=color)
        ax_cam.annotate(
            f"{label} median={median_lat:.1f}ms",
            xy=(t[-1], median_lat),
            fontsize=8,
            color=color,
            va="bottom",
        )
    ax_cam.set_ylabel("Latency (ms)")
    ax_cam.legend(loc="upper left", fontsize=9)
    ax_cam.grid(True, alpha=0.3)

    ax_other = axes[1]
    ax_other.set_title("Joint / Sensor Topics", fontsize=12, fontweight="bold")
    for i, topic in enumerate(sorted(other_topics)):
        t, lat = latencies[topic]
        label = TOPIC_SHORT_NAMES.get(topic, topic.split("/")[-1])
        color = cmap_other(i / max(len(other_topics), 1))
        ax_other.plot(t, lat, linewidth=0.5, alpha=0.7, label=label, color=color)
    ax_other.set_ylabel("Latency (ms)")
    ax_other.set_xlabel("Time (s)")
    ax_other.legend(loc="upper left", fontsize=8, ncol=2)
    ax_other.grid(True, alpha=0.3)

    stats_lines = []
    for topic in sorted(latencies.keys()):
        t, lat = latencies[topic]
        arr = np.asarray(lat)
        name = TOPIC_SHORT_NAMES.get(topic, topic)
        stats_lines.append(
            f"{name:<20s}  n={len(lat):>6d}  "
            f"median={np.median(arr):7.1f}ms  mean={np.mean(arr):7.1f}ms  "
            f"std={np.std(arr):7.1f}ms  min={np.min(arr):7.1f}ms  max={np.max(arr):7.1f}ms"
        )
    stats_text = "\n".join(stats_lines)
    fig.text(
        0.05, -0.01, stats_text, fontsize=7, fontfamily="monospace",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-topic transport latency from a rosbag.")
    parser.add_argument("--input-bag", required=True, help="Path to rosbag2 directory")
    parser.add_argument("--storage-id", default="mcap")
    parser.add_argument("--output", default="plot_topic_latency.png", help="Output PNG path")
    args = parser.parse_args()

    bag_path = str(Path(args.input_bag).resolve())
    bag_name = Path(bag_path).name

    print(f"Reading bag: {bag_path}")
    latencies = collect_latencies(bag_path, args.storage_id)
    print(f"Collected latency data for {len(latencies)} topics")

    for topic in sorted(latencies.keys()):
        t, lat = latencies[topic]
        arr = np.asarray(lat)
        name = TOPIC_SHORT_NAMES.get(topic, topic)
        print(f"  {name:<20s}  n={len(lat):>6d}  median={np.median(arr):.1f}ms  max={np.max(arr):.1f}ms")

    plot_latencies(latencies, args.output, bag_name)


if __name__ == "__main__":
    main()
