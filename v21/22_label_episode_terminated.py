#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 LeRobot v2.1 数据集 episodes.jsonl 增加 episode 级 terminated 标注。

默认只修改 ``meta/episodes.jsonl``，不修改 parquet、videos 或 info.json。

示例:
  python 22_label_episode_terminated.py --dry_run

  python 22_label_episode_terminated.py \
      --dataset_dir /home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0624_nonidle_min30
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path(
    "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0624_nonidle_min30"
)
DEFAULT_TERMINAL_START_EPISODE = 906
DEFAULT_FIELD_NAME = "terminated"
BACKUP_DIR_NAME = "backup_terminal_labels"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="给 LeRobot v2.1 数据集 meta/episodes.jsonl 增加 terminated 标注。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="源 LeRobot v2.1 数据集目录。只修改该目录下的 meta/episodes.jsonl。",
    )
    parser.add_argument(
        "--terminal_start_episode",
        type=int,
        default=DEFAULT_TERMINAL_START_EPISODE,
        help="从该 episode_index 开始标为 terminated=true；之前标为 false。",
    )
    parser.add_argument(
        "--field_name",
        default=DEFAULT_FIELD_NAME,
        help="写入 episodes.jsonl 的字段名。",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印统计，不写文件。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果字段已存在，允许覆盖旧标注。",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="写入前不备份原 episodes.jsonl。",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def validate_field_name(field_name: str) -> None:
    if not field_name:
        raise ValueError("--field_name 不能为空")
    if field_name in {"episode_index", "tasks", "length"}:
        raise ValueError(f"--field_name 不能覆盖 LeRobot 核心字段: {field_name}")


def load_and_validate_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    missing = [path for path in (info_path, episodes_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少必要文件: " + ", ".join(str(path) for path in missing))

    info = read_json(info_path)
    episodes = read_jsonl(episodes_path)
    total_episodes = info.get("total_episodes")
    if not isinstance(total_episodes, int):
        raise ValueError("meta/info.json 缺少有效的 total_episodes")
    if len(episodes) != total_episodes:
        raise ValueError(
            f"episodes.jsonl 条数和 info.json 不一致: rows={len(episodes)}, total_episodes={total_episodes}"
        )

    indices = [int(row.get("episode_index", -1)) for row in episodes]
    expected_indices = list(range(total_episodes))
    if indices != expected_indices:
        raise ValueError("episodes.jsonl 的 episode_index 必须连续为 0..total_episodes-1")

    for row in episodes:
        if "tasks" not in row or "length" not in row:
            raise ValueError(f"episodes.jsonl 行缺少 tasks 或 length: {row}")

    return info, episodes


def ensure_can_write_field(episodes: list[dict[str, Any]], field_name: str, overwrite: bool) -> None:
    existing = [int(row["episode_index"]) for row in episodes if field_name in row]
    if existing and not overwrite:
        preview = ", ".join(str(index) for index in existing[:20])
        if len(existing) > 20:
            preview += f", ... 另有 {len(existing) - 20} 个"
        raise ValueError(
            f"字段 {field_name!r} 已存在于 {len(existing)} 个 episode 中: {preview}。"
            "如需覆盖，请加 --overwrite。"
        )


def label_episodes(
    episodes: list[dict[str, Any]],
    terminal_start_episode: int,
    field_name: str,
) -> list[dict[str, Any]]:
    labeled = []
    for row in episodes:
        new_row = dict(row)
        new_row[field_name] = int(row["episode_index"]) >= terminal_start_episode
        labeled.append(new_row)
    return labeled


def validate_terminal_start(terminal_start_episode: int, total_episodes: int) -> None:
    if terminal_start_episode < 0 or terminal_start_episode > total_episodes:
        raise ValueError(
            f"--terminal_start_episode 必须在 [0, {total_episodes}] 范围内，当前为 {terminal_start_episode}"
        )


def log_summary(
    dataset_dir: Path,
    episodes: list[dict[str, Any]],
    labeled: list[dict[str, Any]],
    terminal_start_episode: int,
    field_name: str,
) -> None:
    total_episodes = len(labeled)
    terminal_count = sum(1 for row in labeled if bool(row[field_name]))
    non_terminal_count = total_episodes - terminal_count
    logger.info("数据集: %s", dataset_dir)
    logger.info("字段名: %s", field_name)
    logger.info("terminal_start_episode: %d", terminal_start_episode)
    logger.info("total_episodes: %d", total_episodes)
    logger.info("%s=false: %d", field_name, non_terminal_count)
    logger.info("%s=true: %d", field_name, terminal_count)
    if episodes:
        sample_indices = sorted(
            {
                0,
                max(0, terminal_start_episode - 1),
                min(total_episodes - 1, terminal_start_episode),
                total_episodes - 1,
            }
        )
        for index in sample_indices:
            row = labeled[index]
            logger.info("样例 episode %d -> %s=%s", index, field_name, row[field_name])


def backup_episodes_file(episodes_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = episodes_path.parent / BACKUP_DIR_NAME
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"episodes.jsonl.{timestamp}.bak"
    shutil.copy2(episodes_path, backup_path)
    return backup_path


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    validate_field_name(args.field_name)

    info, episodes = load_and_validate_dataset(dataset_dir)
    total_episodes = int(info["total_episodes"])
    validate_terminal_start(args.terminal_start_episode, total_episodes)
    ensure_can_write_field(episodes, args.field_name, args.overwrite)

    labeled = label_episodes(episodes, args.terminal_start_episode, args.field_name)
    log_summary(dataset_dir, episodes, labeled, args.terminal_start_episode, args.field_name)

    if args.dry_run:
        logger.info("dry-run 模式：未写入 episodes.jsonl。")
        return

    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not args.no_backup:
        backup_path = backup_episodes_file(episodes_path)
        logger.info("已备份: %s", backup_path)

    write_jsonl(episodes_path, labeled)
    logger.info("已写入: %s", episodes_path)


if __name__ == "__main__":
    main()
