#!/usr/bin/env python3
# ruff: noqa: N999
"""Convert local LeRobot datasets to v2.1, optionally moving embedded images to videos.

The official LeRobot converter is Hub-oriented: it loads ``repo_id`` at the
``v2.0`` revision, computes ``meta/episodes_stats.jsonl``, updates
``meta/info.json``, pushes metadata back to the Hub, deletes ``meta/stats.json``,
and creates a ``v2.1`` tag.

This local variant keeps the same metadata conversion but operates in-place on a
local dataset directory. It never downloads from or uploads to the Hub.

It can also write a new v2.1 dataset directory where parquet-embedded image
features are converted to strict v2.1 video features stored under videos/.

Examples:
    # Convert by direct dataset directory.
    python tools/lerobot_dataset_tools/9_convert_dataset_v20_to_v21.py \
        --dataset_dir /path/to/HF_LEROBOT_HOME/standard/my_dataset

    # Convert by repo_id under HF_LEROBOT_HOME.
    python tools/lerobot_dataset_tools/9_convert_dataset_v20_to_v21.py \
        --repo_id standard/my_dataset \
        --lerobot_home /path/to/HF_LEROBOT_HOME

    # Preview checks without modifying files.
    python tools/lerobot_dataset_tools/9_convert_dataset_v20_to_v21.py \
        --dataset_dir /path/to/dataset \
        --dry_run
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import datasets
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_VIDEO_PATH
from lerobot.common.datasets.utils import EPISODES_STATS_PATH
from lerobot.common.datasets.utils import INFO_PATH
from lerobot.common.datasets.utils import STATS_PATH
from lerobot.common.datasets.utils import load_stats
from lerobot.common.datasets.utils import write_info
from lerobot.common.datasets.v21.convert_stats import check_aggregate_stats
from lerobot.common.datasets.v21.convert_stats import convert_stats
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.datasets.video_utils import get_video_info
from tqdm import tqdm

V20 = "v2.0"
V21 = "v2.1"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SuppressWarnings:
    def __enter__(self):
        self.previous_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.previous_level)


def get_default_lerobot_home() -> Path:
    env_home = os.environ.get("HF_LEROBOT_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local LeRobot datasets to v2.1 and optional videos/.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset_dir",
        type=Path,
        help="Local LeRobot dataset root, for example /data/lerobot/standard/my_dataset",
    )
    group.add_argument(
        "--repo_id",
        type=str,
        help="Dataset repo id under HF_LEROBOT_HOME, for example standard/my_dataset",
    )
    parser.add_argument(
        "--lerobot_home",
        type=Path,
        default=None,
        help="Root containing local LeRobot datasets. Defaults to HF_LEROBOT_HOME or ~/.cache/huggingface/lerobot.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for per-episode stats / image-to-video conversion. Use 0 sequentially.",
    )
    parser.add_argument(
        "--keep_stats",
        action="store_true",
        help="Keep deprecated meta/stats.json after conversion. By default it is backed up and removed.",
    )
    parser.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Back up files that may be replaced or removed into meta/backup_v20_to_v21. Enabled by default.",
    )
    parser.add_argument(
        "--skip_aggregate_check",
        action="store_true",
        help="Skip consistency check between new episodes_stats.jsonl and old stats.json.",
    )
    parser.add_argument(
        "--strict_aggregate_check",
        action="store_true",
        help="Fail the conversion if recomputed episodes_stats do not match old stats.json. Defaults to warning only.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate the local dataset and print the planned changes.",
    )
    parser.add_argument(
        "--images_to_videos",
        action="store_true",
        help="After ensuring v2.1, write a new dataset whose parquet image columns are converted to videos/ mp4.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output dataset directory for --images_to_videos. Required when --images_to_videos is set.",
    )
    parser.add_argument(
        "--image_keys",
        nargs="+",
        default=None,
        help="Image feature keys to convert. Defaults to all dtype=image features.",
    )
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec for --images_to_videos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing output_dir for --images_to_videos.",
    )
    return parser.parse_args()


def resolve_dataset_dir(
    *,
    dataset_dir: Path | None,
    repo_id: str | None,
    lerobot_home: Path | None,
) -> Path:
    if dataset_dir is not None:
        resolved = dataset_dir.expanduser().resolve()
    else:
        if repo_id is None:
            raise ValueError("repo_id is required when dataset_dir is not provided")
        home = lerobot_home.expanduser() if lerobot_home is not None else get_default_lerobot_home()
        resolved = (home / repo_id).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"数据集目录不存在: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"路径不是目录: {resolved}")
    return resolved


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonlines(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            try:
                rows.append(json.loads(stripped_line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
    return rows


def iter_expected_data_files(info: dict, total_episodes: int) -> Iterable[Path]:
    chunks_size = info.get("chunks_size")
    data_path = info.get("data_path")
    if not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError("meta/info.json 缺少有效的 chunks_size")
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("meta/info.json 缺少有效的 data_path")

    for episode_index in range(total_episodes):
        episode_chunk = episode_index // chunks_size
        yield Path(data_path.format(episode_chunk=episode_chunk, episode_index=episode_index))


def iter_expected_video_files(info: dict, total_episodes: int) -> Iterable[Path]:
    chunks_size = info.get("chunks_size")
    video_path = info.get("video_path")
    features = info.get("features", {})
    if not isinstance(video_path, str) or not video_path:
        return
    if not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError("meta/info.json 缺少有效的 chunks_size")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json 缺少有效的 features")

    video_keys = [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    for episode_index in range(total_episodes):
        episode_chunk = episode_index // chunks_size
        for video_key in video_keys:
            yield Path(
                video_path.format(
                    episode_chunk=episode_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )
            )


def get_episode_chunk(info: dict, episode_index: int) -> int:
    chunks_size = info.get("chunks_size")
    if not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError("meta/info.json must contain a positive integer chunks_size")
    return episode_index // chunks_size


def get_data_file_path(info: dict, episode_index: int) -> Path:
    data_path = info.get("data_path")
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("meta/info.json must contain a valid data_path")
    return Path(data_path.format(episode_chunk=get_episode_chunk(info, episode_index), episode_index=episode_index))


def get_video_file_path(info: dict, episode_index: int, video_key: str) -> Path:
    return Path(
        DEFAULT_VIDEO_PATH.format(
            episode_chunk=get_episode_chunk(info, episode_index),
            video_key=video_key,
            episode_index=episode_index,
        )
    )


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def resolve_image_keys(info: dict, image_keys: list[str] | None) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json must contain a features object")

    available = [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "image"
    ]
    if image_keys is None:
        if not available:
            raise ValueError("No dtype=image features found in meta/info.json")
        return available

    missing = [key for key in image_keys if key not in features]
    if missing:
        raise ValueError(f"Requested image_keys are not present in features: {missing}")

    non_images = [key for key in image_keys if features[key].get("dtype") != "image"]
    if non_images:
        raise ValueError(f"Requested image_keys are not dtype=image: {non_images}")

    return list(image_keys)


def plan_image_to_video_conversion(
    dataset_dir: Path,
    output_dir: Path,
    image_keys: list[str] | None = None,
) -> SimpleNamespace:
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    info_path = dataset_dir / INFO_PATH
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"

    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"dataset_dir is not a directory: {dataset_dir}")
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {INFO_PATH}: {info_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing meta/episodes.jsonl: {episodes_path}")

    info = load_json(info_path)
    if info.get("codebase_version") != CODEBASE_VERSION:
        raise ValueError(f"Only LeRobot {CODEBASE_VERSION} datasets are supported, got {info.get('codebase_version')!r}")

    episodes = sorted(load_jsonlines(episodes_path), key=lambda item: item["episode_index"])
    if len(episodes) != info.get("total_episodes"):
        raise ValueError(
            f"episodes.jsonl count does not match total_episodes: {len(episodes)} != {info.get('total_episodes')}"
        )

    resolved_image_keys = resolve_image_keys(info, image_keys)
    for episode in episodes:
        data_file = dataset_dir / get_data_file_path(info, episode["episode_index"])
        if not data_file.is_file():
            raise FileNotFoundError(f"Missing episode parquet file: {data_file}")

    return SimpleNamespace(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        info=info,
        episodes=episodes,
        image_keys=resolved_image_keys,
    )


def copy_dataset(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"output_dir already exists: {dst}. Use --overwrite to replace it.")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def save_episode_frames(dataset: datasets.Dataset, image_key: str, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, row in enumerate(dataset):
        image = row[image_key]
        if image is None:
            raise ValueError(f"Image value is None for key={image_key} frame_index={frame_index}")
        image.convert("RGB").save(frames_dir / f"frame_{frame_index:06d}.png")


def rewrite_parquet_without_image_keys(dataset: datasets.Dataset, parquet_path: Path, image_keys: list[str]) -> None:
    remaining_columns = [column for column in dataset.column_names if column not in image_keys]
    dataset.select_columns(remaining_columns).to_parquet(parquet_path)


def convert_episode_images_to_videos(plan: SimpleNamespace, episode: dict, vcodec: str) -> None:
    episode_index = episode["episode_index"]
    parquet_path = plan.output_dir / get_data_file_path(plan.info, episode_index)
    episode_dataset = datasets.Dataset.from_parquet(str(parquet_path))

    missing_columns = [key for key in plan.image_keys if key not in episode_dataset.column_names]
    if missing_columns:
        raise ValueError(f"{parquet_path} is missing image columns: {missing_columns}")

    with tempfile.TemporaryDirectory(prefix="lerobot_frames_") as temp_dir:
        temp_root = Path(temp_dir)
        for image_key in plan.image_keys:
            frames_dir = temp_root / image_key
            save_episode_frames(episode_dataset, image_key, frames_dir)
            video_path = plan.output_dir / get_video_file_path(plan.info, episode_index, image_key)
            encode_video_frames(frames_dir, video_path, plan.info["fps"], vcodec=vcodec, overwrite=True)

    rewrite_parquet_without_image_keys(episode_dataset, parquet_path, plan.image_keys)


def update_info_for_videos(plan: SimpleNamespace) -> None:
    info = plan.info
    info["video_path"] = DEFAULT_VIDEO_PATH
    info["total_videos"] = info["total_episodes"] * len(plan.image_keys)

    for image_key in plan.image_keys:
        feature = info["features"][image_key]
        feature["dtype"] = "video"
        sample_video = plan.output_dir / get_video_file_path(info, 0, image_key)
        feature["video_info"] = get_video_info(sample_video)
        feature.pop("info", None)

    write_json(info, plan.output_dir / INFO_PATH)


def convert_dataset_images_to_videos(
    dataset_dir: Path,
    output_dir: Path,
    *,
    image_keys: list[str] | None = None,
    num_workers: int = 4,
    vcodec: str = "libsvtav1",
    overwrite: bool = False,
    dry_run: bool = False,
) -> SimpleNamespace:
    plan = plan_image_to_video_conversion(dataset_dir=dataset_dir, output_dir=output_dir, image_keys=image_keys)

    logger.info("image-to-video dataset_dir: %s", plan.dataset_dir)
    logger.info("image-to-video output_dir: %s", plan.output_dir)
    logger.info("image_keys: %s", plan.image_keys)
    logger.info("episodes: %d", len(plan.episodes))
    logger.info("vcodec: %s", vcodec)

    if dry_run:
        return plan

    copy_dataset(plan.dataset_dir, plan.output_dir, overwrite=overwrite)

    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(convert_episode_images_to_videos, plan, episode, vcodec)
                for episode in plan.episodes
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting image columns"):
                future.result()
    else:
        for episode in tqdm(plan.episodes, desc="Converting image columns"):
            convert_episode_images_to_videos(plan, episode, vcodec)

    update_info_for_videos(plan)
    logger.info("image-to-video conversion complete: %s", plan.output_dir)
    return plan


def validate_local_v20_layout(dataset_dir: Path) -> dict:
    dataset_dir = dataset_dir.expanduser().resolve()
    info_path = dataset_dir / INFO_PATH
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    stats_path = dataset_dir / STATS_PATH

    for required_path in (info_path, episodes_path, stats_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"缺少必要文件: {required_path}")

    info = load_json(info_path)
    codebase_version = info.get("codebase_version")
    if codebase_version != V20:
        raise ValueError(f"只支持从 {V20} 转换, 当前 codebase_version={codebase_version!r}")

    episodes = load_jsonlines(episodes_path)
    total_episodes = info.get("total_episodes")
    if not isinstance(total_episodes, int):
        raise ValueError("meta/info.json 缺少有效的 total_episodes")
    if len(episodes) != total_episodes:
        raise ValueError(
            f"episodes.jsonl 条数和 info.json 不一致: "
            f"episodes={len(episodes)} total_episodes={total_episodes}"
        )

    missing_data = [rel_path for rel_path in iter_expected_data_files(info, total_episodes) if not (dataset_dir / rel_path).is_file()]
    if missing_data:
        sample = "\n".join(f"  - {path}" for path in missing_data[:20])
        more = "" if len(missing_data) <= 20 else f"\n  ... 还有 {len(missing_data) - 20} 个未列出"
        raise FileNotFoundError(f"缺失 {len(missing_data)} 个 parquet 文件:\n{sample}{more}")

    missing_videos = [
        rel_path for rel_path in iter_expected_video_files(info, total_episodes) if not (dataset_dir / rel_path).is_file()
    ]
    if missing_videos:
        sample = "\n".join(f"  - {path}" for path in missing_videos[:20])
        more = "" if len(missing_videos) <= 20 else f"\n  ... 还有 {len(missing_videos) - 20} 个未列出"
        raise FileNotFoundError(f"缺失 {len(missing_videos)} 个视频文件:\n{sample}{more}")

    return info


def infer_repo_id(dataset_dir: Path, repo_id: str | None) -> str:
    if repo_id is not None:
        return repo_id
    parent = dataset_dir.parent.name
    if parent:
        return f"{parent}/{dataset_dir.name}"
    return dataset_dir.name


def backup_file(path: Path, backup_dir: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / path.name
    shutil.copy2(path, backup_path)
    return backup_path


def atomic_remove(path: Path, backup_dir: Path | None) -> None:
    if not path.exists():
        return
    if backup_dir is not None:
        backup_file(path, backup_dir)
    path.unlink()


def instantiate_local_dataset(dataset_dir: Path, repo_id: str) -> LeRobotDataset:
    with SuppressWarnings():
        return LeRobotDataset(
            repo_id=repo_id,
            root=dataset_dir,
            revision=V20,
            force_cache_sync=False,
        )


def convert_local_dataset(
    *,
    dataset_dir: Path,
    repo_id: str | None = None,
    num_workers: int = 4,
    keep_stats: bool = False,
    backup: bool = True,
    skip_aggregate_check: bool = False,
    strict_aggregate_check: bool = False,
    dry_run: bool = False,
) -> None:
    dataset_dir = dataset_dir.expanduser().resolve()
    info = validate_local_v20_layout(dataset_dir)
    resolved_repo_id = infer_repo_id(dataset_dir, repo_id)
    backup_dir = dataset_dir / "meta" / "backup_v20_to_v21" if backup else None

    logger.info("数据集目录: %s", dataset_dir)
    logger.info("repo_id: %s", resolved_repo_id)
    logger.info("当前版本: %s -> 目标版本: %s", info["codebase_version"], CODEBASE_VERSION)
    logger.info("episodes: %s", info.get("total_episodes"))
    logger.info("num_workers: %s", num_workers)

    if dry_run:
        logger.info("dry_run: 将重新生成 %s", dataset_dir / EPISODES_STATS_PATH)
        logger.info("dry_run: 将更新 %s 的 codebase_version", dataset_dir / INFO_PATH)
        if keep_stats:
            logger.info("dry_run: 将保留 %s", dataset_dir / STATS_PATH)
        else:
            logger.info("dry_run: 将备份并移除 %s", dataset_dir / STATS_PATH)
        return

    if backup_dir is not None:
        for rel_path in (INFO_PATH, STATS_PATH, EPISODES_STATS_PATH):
            backup_path = backup_file(dataset_dir / rel_path, backup_dir)
            if backup_path is not None:
                logger.info("已备份 %s -> %s", dataset_dir / rel_path, backup_path)

    episodes_stats_path = dataset_dir / EPISODES_STATS_PATH
    if episodes_stats_path.exists():
        episodes_stats_path.unlink()

    dataset = instantiate_local_dataset(dataset_dir, resolved_repo_id)
    convert_stats(dataset, num_workers=num_workers)

    if not skip_aggregate_check:
        ref_stats = load_stats(dataset.root)
        if ref_stats is None:
            raise FileNotFoundError(f"缺少用于一致性检查的旧统计文件: {dataset.root / STATS_PATH}")
        try:
            check_aggregate_stats(dataset, ref_stats)
            logger.info("新 episodes_stats 聚合结果与旧 stats.json 一致")
        except AssertionError as exc:
            if strict_aggregate_check:
                raise
            logger.warning("旧 stats.json 与重新计算的 episodes_stats 不一致, 将继续转换。差异: %s", exc)

    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)

    if keep_stats:
        logger.info("已保留旧统计文件: %s", dataset.root / STATS_PATH)
    else:
        atomic_remove(dataset.root / STATS_PATH, backup_dir)
        logger.info("已移除旧统计文件: %s", dataset.root / STATS_PATH)

    logger.info("转换完成: %s", dataset.root)


def main() -> int:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        lerobot_home=args.lerobot_home,
    )

    info = load_json(dataset_dir / INFO_PATH)
    if info.get("codebase_version") == V20:
        convert_local_dataset(
            dataset_dir=dataset_dir,
            repo_id=args.repo_id,
            num_workers=args.num_workers,
            keep_stats=args.keep_stats,
            backup=args.backup,
            skip_aggregate_check=args.skip_aggregate_check,
            strict_aggregate_check=args.strict_aggregate_check,
            dry_run=args.dry_run,
        )
    elif info.get("codebase_version") == CODEBASE_VERSION:
        logger.info("数据集已经是 %s: %s", CODEBASE_VERSION, dataset_dir)
    else:
        raise ValueError(f"Unsupported codebase_version={info.get('codebase_version')!r}")

    if args.images_to_videos:
        if args.output_dir is None:
            raise ValueError("--output_dir is required when --images_to_videos is set")
        convert_dataset_images_to_videos(
            dataset_dir=dataset_dir,
            output_dir=args.output_dir,
            image_keys=args.image_keys,
            num_workers=args.num_workers,
            vcodec=args.vcodec,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.error("用户中断")
        raise SystemExit(130) from None
    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
