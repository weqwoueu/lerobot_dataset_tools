from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "1_remove_episodes.py"
SPEC = importlib.util.spec_from_file_location("remove_episodes", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
VIDEO_KEYS = ("observation.images.top", "observation.images.wrist")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def formatted_path(template: str, episode_index: int, video_key: str | None = None) -> Path:
    values: dict[str, object] = {
        "episode_chunk": episode_index // 2,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return Path(template.format(**values))


def create_filtered_output(root: Path) -> Path:
    lengths = (2, 3, 1)
    info = {
        "codebase_version": "v2.1",
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 0,
        "total_chunks": 0,
        "total_videos": 0,
        "chunks_size": 2,
        "splits": {"train": f"0:{len(lengths)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {
            **{key: {"dtype": "video", "shape": [2, 2, 3]} for key in VIDEO_KEYS},
            "action": {"dtype": "float32", "shape": [1]},
        },
    }
    write_json(root / "meta" / "info.json", info)
    write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": index, "length": length, "tasks": [f"task-{index % 2}"]}
            for index, length in enumerate(lengths)
        ],
    )
    write_jsonl(
        root / "meta" / "episodes_stats.jsonl",
        [{"episode_index": index, "stats": {}} for index in range(len(lengths))],
    )
    write_jsonl(
        root / "meta" / "tasks.jsonl",
        [
            {"task_index": 0, "task": "task-0"},
            {"task_index": 1, "task": "task-1"},
        ],
    )

    stale_global_starts = (100, 200, 300)
    for episode_index, length in enumerate(lengths):
        parquet_path = root / formatted_path(DATA_PATH, episode_index)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                    "frame_index": pa.array(np.arange(length), type=pa.int64()),
                    "index": pa.array(
                        np.arange(stale_global_starts[episode_index], stale_global_starts[episode_index] + length),
                        type=pa.int64(),
                    ),
                    "task_index": pa.array([episode_index % 2] * length, type=pa.int64()),
                }
            ),
            parquet_path,
        )
        for video_key in VIDEO_KEYS:
            video_path = root / formatted_path(VIDEO_PATH, episode_index, video_key)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"video-placeholder")
    return root


def test_repair_output_info_recomputes_all_derived_fields(tmp_path: Path) -> None:
    output = create_filtered_output(tmp_path / "output")

    repaired = MODULE.repair_output_info(output)

    assert repaired["total_episodes"] == 3
    assert repaired["total_frames"] == 6
    assert repaired["total_tasks"] == 2
    assert repaired["total_chunks"] == 2
    assert repaired["total_videos"] == 6
    assert repaired["splits"] == {"train": "0:3"}
    persisted = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert persisted == repaired
    assert not list((output / "meta").glob(".info.json.*.tmp"))


def test_repair_output_parquet_indices_is_complete_and_idempotent(tmp_path: Path) -> None:
    output = create_filtered_output(tmp_path / "output")

    repaired_count = MODULE.repair_output_parquet_indices(output)

    assert repaired_count == 3
    global_start = 0
    for episode_index, length in enumerate((2, 3, 1)):
        parquet_path = output / formatted_path(DATA_PATH, episode_index)
        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "index"])
        assert table["episode_index"].to_pylist() == [episode_index] * length
        assert table["frame_index"].to_pylist() == list(range(length))
        assert table["index"].to_pylist() == list(range(global_start, global_start + length))
        global_start += length
    assert MODULE.repair_output_parquet_indices(output) == 0
    assert not list((output / "data").rglob("*.tmp"))


def test_repair_output_info_does_not_write_when_video_is_missing(tmp_path: Path) -> None:
    output = create_filtered_output(tmp_path / "output")
    info_path = output / "meta" / "info.json"
    original_content = info_path.read_text(encoding="utf-8")
    (output / formatted_path(VIDEO_PATH, 1, VIDEO_KEYS[0])).unlink()

    with pytest.raises(FileNotFoundError, match="缺少 Parquet/视频文件"):
        MODULE.repair_output_info(output)

    assert info_path.read_text(encoding="utf-8") == original_content


def test_repair_output_info_rejects_discontinuous_episode_stats(tmp_path: Path) -> None:
    output = create_filtered_output(tmp_path / "output")
    stats_path = output / "meta" / "episodes_stats.jsonl"
    stats_rows = [
        {"episode_index": 0, "stats": {}},
        {"episode_index": 1, "stats": {}},
        {"episode_index": 7, "stats": {}},
    ]
    write_jsonl(stats_path, stats_rows)

    with pytest.raises(ValueError, match="必须从 0 连续编号"):
        MODULE.repair_output_info(output)


def test_shared_dataset_writer_reindexes_episode_frame_and_global_indices(tmp_path: Path) -> None:
    hf_dataset = Dataset.from_dict(
        {
            "episode_index": [5, 5, 8],
            "frame_index": [0, 1, 0],
            "index": [100, 101, 900],
            "task_index": [0, 0, 0],
            "value": [1.0, 2.0, 3.0],
        }
    )
    dataset = SimpleNamespace(
        hf_dataset=hf_dataset,
        episode_data_index={"from": [0, 2], "to": [2, 3]},
        meta=SimpleNamespace(info={"chunks_size": 2}),
    )

    dataset_creator_class = MODULE.FilteredDatasetCreator.__mro__[1]
    dataset_creator_class.write_episode_data(dataset, tmp_path, num_episodes=2)

    first = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    second = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000001.parquet")
    assert first["episode_index"].to_pylist() == [0, 0]
    assert first["frame_index"].to_pylist() == [0, 1]
    assert first["index"].to_pylist() == [0, 1]
    assert second["episode_index"].to_pylist() == [1]
    assert second["frame_index"].to_pylist() == [0]
    assert second["index"].to_pylist() == [2]
