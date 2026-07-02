#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""删除长度过短的 LeRobot v2.1 episodes，并生成重新编号的新数据集目录。

默认不会修改源数据集。输出目录默认是源目录同级的 ``<name>_min30``。

示例:
  python 21_remove_short_episodes.py --dry_run

  python 21_remove_short_episodes.py \
      --dataset_dir /home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0624_nonidle
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path(
    "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0624_nonidle"
)
CORE_META_FILES = {"info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="删除 length 小于阈值的 LeRobot v2.1 episodes，并生成新数据集目录。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="源 LeRobot v2.1 数据集目录。脚本只读取该目录，不会原地修改。",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=30,
        help="最小保留长度；删除条件为 episode length < min_length。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="新数据集输出目录。默认是源目录同级的 <name>_min<min_length>。",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="源数据集 repo_id。不传时按 dagger/<dataset_dir.name> 推断。",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将删除/保留的 episodes，不创建输出目录。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已存在的输出目录。",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_dataset_dir(dataset_dir: Path) -> None:
    required_paths = [
        dataset_dir / "meta" / "info.json",
        dataset_dir / "meta" / "episodes.jsonl",
        dataset_dir / "meta" / "episodes_stats.jsonl",
        dataset_dir / "meta" / "tasks.jsonl",
        dataset_dir / "data",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少 LeRobot v2.1 数据集文件/目录: " + ", ".join(str(p) for p in missing))


def infer_repo_id(dataset_dir: Path) -> str:
    return f"dagger/{dataset_dir.name}"


def resolve_output_dir(dataset_dir: Path, output_dir: Path | None, min_length: int) -> Path:
    if output_dir is not None:
        return output_dir
    return dataset_dir.with_name(f"{dataset_dir.name}_min{min_length}")


def get_video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]


def find_short_episodes(episodes: list[dict[str, Any]], min_length: int) -> list[dict[str, int]]:
    short_episodes = []
    for row in episodes:
        if "episode_index" not in row or "length" not in row:
            raise ValueError(f"episodes.jsonl 行缺少 episode_index 或 length: {row}")
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        if length < min_length:
            short_episodes.append({"episode_index": episode_index, "length": length})
    return short_episodes


def build_keep_episodes(episodes: list[dict[str, Any]], short_episodes: list[dict[str, int]]) -> list[int]:
    delete_ids = {item["episode_index"] for item in short_episodes}
    return [int(row["episode_index"]) for row in episodes if int(row["episode_index"]) not in delete_ids]


def log_summary(
    dataset_dir: Path,
    output_dir: Path,
    repo_id: str,
    episodes: list[dict[str, Any]],
    short_episodes: list[dict[str, int]],
    min_length: int,
) -> None:
    total_frames = sum(int(row["length"]) for row in episodes)
    removed_frames = sum(item["length"] for item in short_episodes)
    logger.info("源数据集: %s", dataset_dir)
    logger.info("输出目录: %s", output_dir)
    logger.info("repo_id: %s", repo_id)
    logger.info("删除条件: length < %d", min_length)
    logger.info("episodes: %d -> %d", len(episodes), len(episodes) - len(short_episodes))
    logger.info("frames: %d -> %d", total_frames, total_frames - removed_frames)
    logger.info("将删除 episodes: %d", len(short_episodes))
    if short_episodes:
        preview = ", ".join(f"{item['episode_index']}({item['length']})" for item in short_episodes[:50])
        if len(short_episodes) > 50:
            preview += f", ... 另有 {len(short_episodes) - 50} 个"
        logger.info("待删 episode(length): %s", preview)


def load_creator_dependencies():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from dataset_creator.filtered_dataset_creator import FilteredDatasetCreator

        return LeRobotDataset, FilteredDatasetCreator
    except ImportError:
        current_dir = Path(__file__).resolve().parent
        if str(current_dir) not in sys.path:
            sys.path.append(str(current_dir))
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from dataset_creator.filtered_dataset_creator import FilteredDatasetCreator

            return LeRobotDataset, FilteredDatasetCreator
        except ImportError as exc:
            raise RuntimeError(
                "无法导入 lerobot 或 dataset_creator，请在 v21 的 LeRobot 处理环境中运行。"
            ) from exc


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        return
    if not force:
        raise FileExistsError(f"输出目录已存在，请换一个路径或加 --force: {output_dir}")
    logger.warning("删除已存在的输出目录: %s", output_dir)
    shutil.rmtree(output_dir)


def copy_auxiliary_meta_files(dataset_dir: Path, output_dir: Path) -> None:
    src_meta_dir = dataset_dir / "meta"
    dst_meta_dir = output_dir / "meta"
    if not src_meta_dir.is_dir() or not dst_meta_dir.is_dir():
        return
    copied = []
    for src_path in sorted(src_meta_dir.iterdir()):
        if not src_path.is_file() or src_path.name in CORE_META_FILES:
            continue
        dst_path = dst_meta_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        copied.append(src_path.name)
    if copied:
        logger.info("已复制辅助 meta 文件: %s", ", ".join(copied))


def patch_output_info(output_dir: Path) -> None:
    info_path = output_dir / "meta" / "info.json"
    episodes_path = output_dir / "meta" / "episodes.jsonl"
    info = read_json(info_path)
    episodes = read_jsonl(episodes_path)
    total_episodes = len(episodes)
    chunks_size = int(info["chunks_size"])
    video_keys = get_video_keys(info)

    info["total_episodes"] = total_episodes
    info["total_frames"] = sum(int(row["length"]) for row in episodes)
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["total_chunks"] = (total_episodes - 1) // chunks_size + 1 if total_episodes else 0
    info["total_videos"] = total_episodes * len(video_keys) if info.get("video_path") else 0
    write_json(info_path, info)


def create_filtered_dataset(
    dataset_dir: Path,
    output_dir: Path,
    repo_id: str,
    selected_episodes: list[int],
    min_length: int,
) -> None:
    LeRobotDataset, FilteredDatasetCreator = load_creator_dependencies()
    new_repo_id = f"{repo_id}_min{min_length}"

    logger.info("加载源数据集: repo_id=%s, root=%s", repo_id, dataset_dir)
    dataset = LeRobotDataset(repo_id, root=dataset_dir)

    creator = FilteredDatasetCreator(original_dataset=dataset)
    success = creator.create(
        new_repo_id=new_repo_id,
        selected_episodes=selected_episodes,
        push_to_hub=False,
        local_output_dir=output_dir,
        temp_dir=output_dir.parent,
    )
    if not success:
        raise RuntimeError("FilteredDatasetCreator 创建数据集失败")

    copy_auxiliary_meta_files(dataset_dir, output_dir)
    patch_output_info(output_dir)


def validate_output(output_dir: Path, min_length: int) -> None:
    info = read_json(output_dir / "meta" / "info.json")
    episodes = read_jsonl(output_dir / "meta" / "episodes.jsonl")
    indices = [int(row["episode_index"]) for row in episodes]
    expected_indices = list(range(len(episodes)))
    if indices != expected_indices:
        raise ValueError("输出 episodes.jsonl 的 episode_index 不连续")
    short_episodes = find_short_episodes(episodes, min_length)
    if short_episodes:
        raise ValueError(f"输出数据集中仍存在 length < {min_length} 的 episode: {short_episodes}")
    if int(info["total_episodes"]) != len(episodes):
        raise ValueError("输出 info.json total_episodes 与 episodes.jsonl 条数不一致")
    if int(info["total_frames"]) != sum(int(row["length"]) for row in episodes):
        raise ValueError("输出 info.json total_frames 与 episodes.jsonl length 总和不一致")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = resolve_output_dir(dataset_dir, args.output_dir, args.min_length).resolve()
    repo_id = args.repo_id or infer_repo_id(dataset_dir)

    if args.min_length <= 0:
        raise ValueError("--min_length 必须大于 0")

    validate_dataset_dir(dataset_dir)
    info = read_json(dataset_dir / "meta" / "info.json")
    episodes = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(
            f"episodes.jsonl 条数和 info.json 不一致: {len(episodes)} != {info['total_episodes']}"
        )

    short_episodes = find_short_episodes(episodes, args.min_length)
    selected_episodes = build_keep_episodes(episodes, short_episodes)
    log_summary(dataset_dir, output_dir, repo_id, episodes, short_episodes, args.min_length)

    if not short_episodes:
        logger.info("没有发现需要删除的短 episode，程序结束。")
        return
    if not selected_episodes:
        raise ValueError("删除后没有剩余 episode，操作终止。")
    if args.dry_run:
        logger.info("dry-run 模式：未创建输出目录。")
        return

    prepare_output_dir(output_dir, args.force)
    create_filtered_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        repo_id=repo_id,
        selected_episodes=selected_episodes,
        min_length=args.min_length,
    )
    validate_output(output_dir, args.min_length)
    logger.info("完成: %s", output_dir)


if __name__ == "__main__":
    main()
