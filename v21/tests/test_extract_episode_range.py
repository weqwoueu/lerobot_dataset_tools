from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "26_extract_episode_range.py"
SPEC = importlib.util.spec_from_file_location("extract_episode_range", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


LENGTHS = (2, 3, 2, 2, 1)
TASKS_BY_EPISODE = (0, 2, 2, 1, 0)
TASK_NAMES = ("alpha", "beta", "gamma")
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
VIDEO_KEY = "observation.images.top"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def dataset_path(template: str, episode_index: int, video_key: str | None = None) -> Path:
    values: dict[str, object] = {
        "episode_chunk": episode_index // 2,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return Path(template.format(**values))


def episode_stats(episode_index: int, length: int) -> dict[str, object]:
    value = float(episode_index + 1)
    return {
        "episode_index": episode_index,
        "stats": {
            "observation.state": {
                "min": [value],
                "max": [value],
                "mean": [value],
                "std": [0.0],
                "count": [length],
            }
        },
    }


def create_source_dataset(
    root: Path,
    *,
    include_videos: bool = True,
    info_overrides: dict[str, object] | None = None,
) -> Path:
    features: dict[str, object] = {
        "observation.state": {"dtype": "float32", "shape": [1], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [1], "names": ["action"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if include_videos:
        features[VIDEO_KEY] = {
            "dtype": "video",
            "shape": [3, 2, 2],
            "names": ["channels", "height", "width"],
            "info": {"video.fps": 10.0, "video.codec": "h264"},
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "total_episodes": len(LENGTHS),
        "total_frames": sum(LENGTHS),
        "total_tasks": len(TASK_NAMES),
        "total_videos": len(LENGTHS) if include_videos else 0,
        "total_chunks": 3,
        "chunks_size": 2,
        "fps": 10,
        "splits": {"train": f"0:{len(LENGTHS)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH if include_videos else None,
        "features": features,
    }
    if info_overrides is not None:
        info.update(info_overrides)
    write_json(root / "meta" / "info.json", info)
    write_jsonl(
        root / "meta" / "tasks.jsonl",
        [{"task_index": index, "task": name} for index, name in enumerate(TASK_NAMES)],
    )

    episodes = []
    stats = []
    global_index = 0
    for episode_index, (length, task_index) in enumerate(zip(LENGTHS, TASKS_BY_EPISODE, strict=True)):
        task_name = TASK_NAMES[task_index]
        episodes.append({"episode_index": episode_index, "tasks": [task_name], "length": length})
        stats.append(episode_stats(episode_index, length))

        frame_indices = np.arange(length, dtype=np.int64)
        table = pa.table(
            {
                "observation.state": pa.array(
                    [[float(episode_index + 1)]] * length,
                    type=pa.list_(pa.float32(), 1),
                ),
                "action": pa.array(
                    [[float(episode_index + 10)]] * length,
                    type=pa.list_(pa.float32(), 1),
                ),
                "timestamp": pa.array(frame_indices / 10.0, type=pa.float32()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                "index": pa.array(
                    np.arange(global_index, global_index + length, dtype=np.int64),
                    type=pa.int64(),
                ),
                "task_index": pa.array([task_index] * length, type=pa.int64()),
            }
        )
        parquet_path = root / dataset_path(DATA_PATH, episode_index)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)

        if include_videos:
            video_path = root / dataset_path(VIDEO_PATH, episode_index, VIDEO_KEY)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(f"video-{episode_index}".encode())
        global_index += length

    write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    write_jsonl(root / "meta" / "episodes_stats.jsonl", stats)
    write_json(root / "meta" / "stats.json", {"stale": True})
    write_json(root / "norm_stats.json", {"stale": True})
    (root / "README.md").write_text("source readme\n", encoding="utf-8")
    (root / "old.backup.json").write_text("{}\n", encoding="utf-8")
    (root / "meta" / "lineage.json").write_text("{}\n", encoding="utf-8")
    (root / "meta" / "note.txt").write_text("keep me\n", encoding="utf-8")
    return root


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_extract_closed_range_reindexes_metadata_parquet_tasks_and_videos(tmp_path: Path) -> None:
    source = create_source_dataset(tmp_path / "source")
    output = tmp_path / "output"

    actual_output = MODULE.process_dataset(source, 1, 3, output)

    assert actual_output == output
    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 7
    assert info["total_tasks"] == 2
    assert info["total_chunks"] == 2
    assert info["total_videos"] == 3
    assert info["splits"] == {"train": "0:3"}
    assert info["data_path"] == DATA_PATH
    assert info["video_path"] == VIDEO_PATH

    episodes = load_jsonl(output / "meta" / "episodes.jsonl")
    assert [row["episode_index"] for row in episodes] == [0, 1, 2]
    assert [row["length"] for row in episodes] == [3, 2, 2]
    assert [row["tasks"] for row in episodes] == [["gamma"], ["gamma"], ["beta"]]

    stats = load_jsonl(output / "meta" / "episodes_stats.jsonl")
    assert [row["episode_index"] for row in stats] == [0, 1, 2]
    assert [row["stats"]["observation.state"]["mean"] for row in stats] == [[2.0], [3.0], [4.0]]

    tasks = load_jsonl(output / "meta" / "tasks.jsonl")
    assert tasks == [
        {"task_index": 0, "task": "beta"},
        {"task_index": 1, "task": "gamma"},
    ]

    expected_global_start = 0
    for destination_episode, source_episode in enumerate(range(1, 4)):
        source_table = pq.read_table(source / dataset_path(DATA_PATH, source_episode))
        output_table = pq.read_table(output / dataset_path(DATA_PATH, destination_episode))
        length = LENGTHS[source_episode]
        assert output_table.schema == source_table.schema
        assert output_table["observation.state"].equals(source_table["observation.state"])
        assert output_table["action"].equals(source_table["action"])
        assert output_table["frame_index"].equals(source_table["frame_index"])
        assert output_table["timestamp"].equals(source_table["timestamp"])
        assert output_table["episode_index"].to_pylist() == [destination_episode] * length
        assert output_table["index"].to_pylist() == list(
            range(expected_global_start, expected_global_start + length)
        )
        expected_task_index = 1 if source_episode in {1, 2} else 0
        assert output_table["task_index"].to_pylist() == [expected_task_index] * length

        source_video = source / dataset_path(VIDEO_PATH, source_episode, VIDEO_KEY)
        output_video = output / dataset_path(VIDEO_PATH, destination_episode, VIDEO_KEY)
        assert output_video.read_bytes() == source_video.read_bytes()
        expected_global_start += length

    aggregate_stats = json.loads((output / "meta" / "stats.json").read_text(encoding="utf-8"))
    expected_mean = (2.0 * 3 + 3.0 * 2 + 4.0 * 2) / 7
    assert aggregate_stats["observation.state"]["mean"] == pytest.approx([expected_mean])
    assert aggregate_stats["observation.state"]["count"] == [7]
    assert aggregate_stats["observation.state"]["min"] == [2.0]
    assert aggregate_stats["observation.state"]["max"] == [4.0]

    assert (output / "README.md").read_text(encoding="utf-8") == "source readme\n"
    assert (output / "meta" / "note.txt").read_text(encoding="utf-8") == "keep me\n"
    assert not (output / "norm_stats.json").exists()
    assert not (output / "old.backup.json").exists()
    assert not (output / "meta" / "lineage.json").exists()


def test_stale_source_info_derived_fields_do_not_leak_to_output(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = create_source_dataset(
        tmp_path / "source",
        info_overrides={
            "total_episodes": 999,
            "total_frames": 999,
            "total_tasks": 999,
            "total_chunks": 0,
            "total_videos": 0,
            "splits": {"train": "0:0"},
        },
    )
    output = tmp_path / "output"
    caplog.set_level("WARNING", logger=MODULE.logger.name)

    MODULE.process_dataset(source, 1, 3, output)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "total_chunks" in messages
    assert "total_videos" in messages
    assert "splits" in messages

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 7
    assert info["total_tasks"] == 2
    assert info["total_chunks"] == 2
    assert info["total_videos"] == 3
    assert info["splits"] == {"train": "0:3"}


def test_single_episode_and_default_output_name(tmp_path: Path) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)

    output = MODULE.process_dataset(source, 4, 4)

    assert output == tmp_path / "source_ep4_4"
    table = pq.read_table(output / dataset_path(DATA_PATH, 0))
    assert table["episode_index"].to_pylist() == [0]
    assert table["index"].to_pylist() == [0]
    assert json.loads((output / "meta" / "info.json").read_text())["total_videos"] == 0


def test_full_range_preserves_counts_while_republishing_dataset(tmp_path: Path) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)

    output = MODULE.process_dataset(source, 0, len(LENGTHS) - 1, tmp_path / "output")

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == len(LENGTHS)
    assert info["total_frames"] == sum(LENGTHS)
    assert info["total_tasks"] == len(TASK_NAMES)
    assert len(list((output / "data").rglob("*.parquet"))) == len(LENGTHS)


def test_dry_run_validates_but_does_not_touch_existing_output(tmp_path: Path) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("original", encoding="utf-8")

    MODULE.process_dataset(source, 0, 1, output, dry_run=True)

    assert marker.read_text(encoding="utf-8") == "original"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize(
    ("start_episode", "end_episode", "message"),
    [
        (-1, 1, "不能为负数"),
        (2, 1, "不能大于"),
        (0, 5, "范围越界"),
    ],
)
def test_invalid_ranges_fail_without_creating_output(
    tmp_path: Path,
    start_episode: int,
    end_episode: int,
    message: str,
) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match=message):
        MODULE.process_dataset(source, start_episode, end_episode, output)

    assert not output.exists()


def test_discontinuous_source_episode_metadata_is_rejected(tmp_path: Path) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)
    episodes_path = source / "meta" / "episodes.jsonl"
    episodes = load_jsonl(episodes_path)
    episodes[2]["episode_index"] = 7
    write_jsonl(episodes_path, episodes)

    with pytest.raises(ValueError, match="必须从 0 连续编号"):
        MODULE.process_dataset(source, 1, 2, tmp_path / "output")


def test_missing_video_is_rejected_before_output_is_created(tmp_path: Path) -> None:
    source = create_source_dataset(tmp_path / "source")
    missing_video = source / dataset_path(VIDEO_PATH, 2, VIDEO_KEY)
    missing_video.unlink()
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="源视频不存在"):
        MODULE.process_dataset(source, 1, 2, output)

    assert not output.exists()


def test_missing_parquet_is_rejected_before_output_is_created(tmp_path: Path) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)
    missing_parquet = source / dataset_path(DATA_PATH, 2)
    missing_parquet.unlink()
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="源 Parquet 不存在"):
        MODULE.process_dataset(source, 1, 2, output)

    assert not output.exists()


def test_parquet_length_mismatch_is_rejected_before_output_is_created(tmp_path: Path) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)
    parquet_path = source / dataset_path(DATA_PATH, 2)
    table = pq.read_table(parquet_path).slice(0, 1)
    pq.write_table(table, parquet_path)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="行数和 episodes.jsonl 不一致"):
        MODULE.process_dataset(source, 2, 2, output)

    assert not output.exists()


def test_existing_output_requires_force(tmp_path: Path) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError, match="--force"):
        MODULE.process_dataset(source, 0, 0, output)


def test_force_failure_preserves_existing_output_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("original", encoding="utf-8")

    def fail_validation(_plan: object, _staging_dir: Path) -> None:
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(MODULE, "validate_output_dataset", fail_validation)

    with pytest.raises(RuntimeError, match="injected validation failure"):
        MODULE.process_dataset(source, 1, 2, output, force=True)

    assert marker.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".output.extract-*"))
    assert not list(tmp_path.glob(".output.backup-*"))


def test_force_success_replaces_existing_output(tmp_path: Path) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)
    output = tmp_path / "output"
    output.mkdir()
    (output / "marker.txt").write_text("old", encoding="utf-8")

    MODULE.process_dataset(source, 0, 0, output, force=True)

    assert not (output / "marker.txt").exists()
    assert (output / "meta" / "info.json").is_file()
    assert not list(tmp_path.glob(".output.backup-*"))


def test_force_publish_failure_rolls_back_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = create_source_dataset(tmp_path / "source", include_videos=False)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("original", encoding="utf-8")
    real_replace = MODULE.os.replace

    def fail_staging_publish(source_path: str | Path, destination_path: str | Path) -> None:
        source_candidate = Path(source_path)
        destination_candidate = Path(destination_path)
        if source_candidate.name.startswith(".output.extract-") and destination_candidate == output:
            raise OSError("injected publish failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(MODULE.os, "replace", fail_staging_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        MODULE.process_dataset(source, 1, 2, output, force=True)

    assert marker.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".output.extract-*"))
    assert not list(tmp_path.glob(".output.backup-*"))


def test_cli_accepts_hyphen_and_underscore_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--dataset_dir",
            "/tmp/source",
            "--start-episode",
            "1",
            "--end_episode",
            "3",
            "--output_dir",
            "/tmp/output",
            "--dry_run",
        ],
    )

    args = MODULE.parse_args()

    assert args.dataset_dir == Path("/tmp/source")
    assert args.start_episode == 1
    assert args.end_episode == 3
    assert args.output_dir == Path("/tmp/output")
    assert args.dry_run is True


def test_output_can_be_loaded_by_lerobot_v21(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "huggingface_datasets_cache"
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_dir))
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(cache_dir))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = create_source_dataset(tmp_path / "source", include_videos=False)
    output = MODULE.process_dataset(source, 1, 3, tmp_path / "output")

    dataset = LeRobotDataset("local/extracted", root=output)

    assert dataset.num_episodes == 3
    assert dataset.num_frames == 7
    assert dataset.meta.total_tasks == 2
    assert dataset[0]["episode_index"].item() == 0
    assert dataset[0]["index"].item() == 0
    assert dataset[0]["task"] == "gamma"
