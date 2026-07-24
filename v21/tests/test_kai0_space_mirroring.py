from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "13_kai0_space_mirroring.py"
SPEC = importlib.util.spec_from_file_location("kai0_space_mirroring", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

FILTER_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "11_filter_nonidle_frames.py"
FILTER_SPEC = importlib.util.spec_from_file_location(
    "filter_nonidle_frames_for_space_mirror_test", FILTER_SCRIPT_PATH
)
assert FILTER_SPEC is not None and FILTER_SPEC.loader is not None
FILTER_MODULE = importlib.util.module_from_spec(FILTER_SPEC)
sys.modules[FILTER_SPEC.name] = FILTER_MODULE
FILTER_SPEC.loader.exec_module(FILTER_MODULE)


def make_config(
    negate_joints: tuple[int, ...] = (1, 4, 6),
) -> MODULE.MirrorConfig:
    return MODULE.MirrorConfig(
        state_key="observation.state",
        action_key="action",
        negate_joints=negate_joints,
    )


def write_info(root: Path, features: dict[str, object]) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(
        json.dumps({"features": features}), encoding="utf-8"
    )


def test_piper_joint_mapping_swaps_arms_and_negates_both_sides() -> None:
    values = np.arange(1, 15, dtype=np.float64)

    actual = MODULE.swap_arms_in_array(values, make_config())

    expected = np.asarray(
        [-8, 9, 10, -11, 12, -13, 14, -1, 2, 3, -4, 5, -6, 7],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(actual, expected)


def test_piperx_joint_mapping_uses_axes_1_5_6() -> None:
    values = np.arange(1, 15, dtype=np.float64)

    actual = MODULE.swap_arms_in_array(values, make_config((1, 5, 6)))

    expected = np.asarray(
        [-8, 9, 10, 11, -12, -13, 14, -1, 2, 3, 4, -5, -6, 7],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(actual, expected)


def test_stats_transform_handles_mean_std_and_bounds_correctly() -> None:
    config = make_config()
    stats = {
        "mean": list(range(1, 15)),
        "std": list(range(21, 35)),
        "min": list(range(1, 15)),
        "max": list(range(101, 115)),
    }

    MODULE.transform_stats_item(stats, config, (("min", "max"),))

    assert stats["mean"] == [-8, 9, 10, -11, 12, -13, 14, -1, 2, 3, -4, 5, -6, 7]
    assert stats["std"] == list(range(28, 35)) + list(range(21, 28))
    assert stats["min"] == [
        -108,
        9,
        10,
        -111,
        12,
        -113,
        14,
        -101,
        2,
        3,
        -104,
        5,
        -106,
        7,
    ]
    assert stats["max"] == [
        -8,
        109,
        110,
        -11,
        112,
        -13,
        114,
        -1,
        102,
        103,
        -4,
        105,
        -6,
        107,
    ]


def test_quantile_pair_must_be_complete() -> None:
    with pytest.raises(ValueError, match="q01/q99 必须成对存在"):
        MODULE.transform_stats_item(
            {"q01": list(range(14))}, make_config(), (("q01", "q99"),)
        )


def test_norm_stats_quantiles_swap_bounds_when_negated(tmp_path: Path) -> None:
    source_path = tmp_path / "norm_stats.json"
    output_path = tmp_path / "output" / "norm_stats.json"
    stat_item = {
        "mean": list(range(1, 15)),
        "std": list(range(21, 35)),
        "q01": list(range(1, 15)),
        "q99": list(range(101, 115)),
    }
    source_path.write_text(
        json.dumps({"norm_stats": {"state": stat_item, "actions": stat_item}}),
        encoding="utf-8",
    )
    config = MODULE.MirrorConfig(
        state_key="observation.state",
        action_key="action",
        negate_joints=(1,),
        norm_state_key="state",
        norm_action_key="actions",
    )

    _, success, _ = MODULE.process_norm_stats_json(source_path, output_path, config)

    assert success
    output = json.loads(output_path.read_text(encoding="utf-8"))["norm_stats"]
    assert output["state"]["q01"][0] == -108
    assert output["state"]["q99"][0] == -8
    assert output["actions"]["q01"][7] == -101
    assert output["actions"]["q99"][7] == -1


def test_parquet_transform_uses_custom_keys(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "output.parquet"
    values = np.arange(1, 15, dtype=np.float64)
    pd.DataFrame(
        {
            "robot_state": [values],
            "robot_action": [values * 10],
            "frame_index": [0],
        }
    ).to_parquet(source_path, index=False)
    config = MODULE.MirrorConfig(
        state_key="robot_state",
        action_key="robot_action",
        negate_joints=(1, 4, 6),
    )

    _, success, message = MODULE.swap_arms_in_parquet(source_path, output_path, config)

    assert success, message
    output = pd.read_parquet(output_path)
    assert output.loc[0, "robot_state"].tolist() == [
        -8,
        9,
        10,
        -11,
        12,
        -13,
        14,
        -1,
        2,
        3,
        -4,
        5,
        -6,
        7,
    ]
    assert output.loc[0, "robot_action"][0] == -80
    assert output.loc[0, "frame_index"] == 0


def test_episode_stats_uses_resolved_keys_and_swaps_single_hand_camera() -> None:
    config = MODULE.MirrorConfig(
        state_key="state",
        action_key="actions",
        negate_joints=(1,),
    )
    vector_stats = {
        "mean": list(range(1, 15)),
        "std": list(range(1, 15)),
        "min": list(range(1, 15)),
        "max": list(range(101, 115)),
    }
    stats = {
        "state": json.loads(json.dumps(vector_stats)),
        "actions": json.loads(json.dumps(vector_stats)),
        "observation.images.hand_left": {"mean": [0.1]},
    }

    actual = MODULE.swap_stats_dims(stats, config)

    assert "observation.images.hand_left" not in actual
    assert actual["observation.images.hand_right"] == {"mean": [0.1]}
    assert actual["state"]["mean"][0] == -8
    assert actual["actions"]["mean"][7] == -1


def test_episode_stats_swaps_cam_wrist_camera_stats() -> None:
    config = MODULE.MirrorConfig(
        state_key="state",
        action_key="action",
        negate_joints=(1,),
    )
    vector_stats = {
        "mean": list(range(1, 15)),
        "std": list(range(1, 15)),
        "min": list(range(1, 15)),
        "max": list(range(101, 115)),
    }
    stats = {
        "state": json.loads(json.dumps(vector_stats)),
        "action": json.loads(json.dumps(vector_stats)),
        "observation.images.cam_left_wrist": {"mean": [0.1]},
        "observation.images.cam_right_wrist": {"mean": [0.9]},
        "observation.images.cam_high": {"mean": [0.5]},
    }

    actual = MODULE.swap_stats_dims(stats, config)

    assert actual["observation.images.cam_left_wrist"] == {"mean": [0.9]}
    assert actual["observation.images.cam_right_wrist"] == {"mean": [0.1]}
    assert actual["observation.images.cam_high"] == {"mean": [0.5]}


def test_resolve_config_auto_detects_feature_and_legacy_norm_keys(tmp_path: Path) -> None:
    write_info(
        tmp_path,
        {
            "observation.state": {"shape": [14]},
            "action": {"shape": [14]},
        },
    )
    (tmp_path / "norm_stats.json").write_text(
        json.dumps({"norm_stats": {"state": {}, "actions": {}}}),
        encoding="utf-8",
    )

    config = MODULE.resolve_mirror_config(tmp_path, None, None, [6, 1, 4])

    assert config.state_key == "observation.state"
    assert config.action_key == "action"
    assert config.state_key_source == "meta/info.json 自动识别"
    assert config.action_key_source == "meta/info.json 自动识别"
    assert config.norm_state_key == "state"
    assert config.norm_action_key == "actions"
    assert config.negate_joints == (1, 4, 6)


def test_explicit_custom_keys_override_auto_detection(tmp_path: Path) -> None:
    write_info(
        tmp_path,
        {
            "robot_state": {"shape": [14]},
            "robot_action": {"shape": [14]},
        },
    )

    config = MODULE.resolve_mirror_config(
        tmp_path, "robot_state", "robot_action", [1, 5, 6]
    )

    assert config.state_key == "robot_state"
    assert config.action_key == "robot_action"
    assert config.state_key_source == "命令行参数"
    assert config.action_key_source == "命令行参数"


def test_auto_detection_rejects_ambiguous_state_keys(tmp_path: Path) -> None:
    write_info(
        tmp_path,
        {
            "observation.state": {"shape": [14]},
            "state": {"shape": [14]},
            "action": {"shape": [14]},
        },
    )

    with pytest.raises(ValueError, match="多个候选 state key"):
        MODULE.resolve_mirror_config(tmp_path, None, None, [1])


def test_signal_feature_must_be_14_dimensions(tmp_path: Path) -> None:
    write_info(
        tmp_path,
        {
            "observation.state": {"shape": [13]},
            "action": {"shape": [14]},
        },
    )

    with pytest.raises(ValueError, match="仅支持 14 维双臂 state"):
        MODULE.resolve_mirror_config(tmp_path, None, None, [1])


@pytest.mark.parametrize(
    ("left_dim", "right_dim", "joints", "message"),
    [
        (6, 7, [1], "仅支持左右臂各 7 维"),
        (7, 7, [], "必须通过 --negate-joints"),
        (7, 7, [1, 1], "不能包含重复轴"),
        (7, 7, [7], "只能包含 1..6"),
    ],
)
def test_mirror_options_are_strictly_validated(
    left_dim: int,
    right_dim: int,
    joints: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE.validate_mirror_options(left_dim, right_dim, joints)


def test_printed_dimension_mapping_comes_from_config(capsys: pytest.CaptureFixture[str]) -> None:
    config = make_config()

    MODULE.print_conversion_plan(Path("/source"), Path("/mirror"), config)

    output = capsys.readouterr().out
    assert "State key: observation.state" in output
    assert "Action key: action" in output
    assert "取反关节（从 1 开始）: [1, 4, 6]" in output
    assert "输出[ 1] 左臂关节1" in output
    assert "= -输入[ 8] 右臂关节1" in output
    assert "输出[ 7] 左臂夹爪" in output
    assert "= +输入[14] 右臂夹爪" in output


def test_create_mirror_dataset_runs_with_auto_detected_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "mirror"
    write_info(
        source_dir,
        {
            "observation.state": {"shape": [14]},
            "action": {"shape": [14]},
        },
    )
    (source_dir / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
    (source_dir / "meta" / "tasks.jsonl").write_text("{}\n", encoding="utf-8")
    vector_stats = {
        "mean": list(range(1, 15)),
        "std": list(range(1, 15)),
        "min": list(range(1, 15)),
        "max": list(range(101, 115)),
    }
    episode_stats = {
        "stats": {
            "observation.state": vector_stats,
            "action": vector_stats,
        }
    }
    (source_dir / "meta" / "episodes_stats.jsonl").write_text(
        json.dumps(episode_stats) + "\n", encoding="utf-8"
    )
    parquet_dir = source_dir / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True)
    values = np.arange(1, 15, dtype=np.float64)
    pd.DataFrame(
        {"observation.state": [values], "action": [values], "frame_index": [0]}
    ).to_parquet(parquet_dir / "episode_000000.parquet", index=False)

    MODULE.create_mirror_dataset(
        src_path=str(source_dir),
        tgt_path=str(output_dir),
        num_workers=1,
        negate_joints=[1, 4, 6],
    )

    output = pd.read_parquet(output_dir / "data" / "chunk-000" / "episode_000000.parquet")
    assert output.loc[0, "observation.state"][0] == -8
    assert output.loc[0, "action"][13] == 7
    assert (output_dir / "meta" / "info.json").is_file()
    console = capsys.readouterr().out
    assert console.index("空间镜像转换方案") < console.index("Starting to create mirrored dataset")


def make_video_params(
    codec_name: str = "h264",
    width: int = 640,
    height: int = 480,
    pix_fmt: str = "yuv420p",
) -> MODULE.VideoParams:
    return MODULE.VideoParams(
        codec_name=codec_name,
        codec_tag="avc1",
        pix_fmt=pix_fmt,
        width=width,
        height=height,
        fps=30.0,
        fps_expr="30/1",
        gop=30,
        b_frames=2,
    )


def filter_video_args(config: MODULE.VideoEncodeConfig) -> argparse.Namespace:
    return argparse.Namespace(
        video_codec=config.video_codec,
        gop=config.gop,
        b_frames=config.b_frames,
        nvenc_preset=config.nvenc_preset,
        nvenc_cq=config.nvenc_cq,
        av1_crf=config.av1_crf,
        av1_cpu_used=config.av1_cpu_used,
        mp4v_qscale=config.mp4v_qscale,
    )


@pytest.mark.parametrize(
    ("video_codec", "source_codec", "width", "height"),
    [
        ("h264_nvenc", "h264", 640, 480),
        ("h264_nvenc", "h264", 128, 128),
        ("av1", "h264", 640, 480),
        ("mp4v", "h264", 640, 480),
        ("source", "h264", 640, 480),
    ],
)
def test_video_encoding_options_match_filter_nonidle_frames(
    video_codec: str,
    source_codec: str,
    width: int,
    height: int,
) -> None:
    config = MODULE.VideoEncodeConfig(video_codec=video_codec)
    params = make_video_params(source_codec, width, height)
    filter_params = FILTER_MODULE.VideoParams(
        codec_name=params.codec_name,
        codec_tag=params.codec_tag,
        pix_fmt=params.pix_fmt,
        width=params.width,
        height=params.height,
        fps=params.fps,
        fps_expr=params.fps_expr,
        gop=params.gop,
        b_frames=params.b_frames,
    )
    filter_args = filter_video_args(config)

    encoding = MODULE.resolve_video_encoding(params, config)

    assert encoding.codec == FILTER_MODULE.output_codec_name(filter_params, filter_args)
    assert list(encoding.codec_args) == FILTER_MODULE.ffmpeg_codec_args(
        filter_params, filter_args
    )
    assert encoding.pix_fmt == FILTER_MODULE.output_pix_fmt(
        filter_params, encoding.codec
    )
    assert encoding.gop == FILTER_MODULE.output_gop(filter_params, filter_args)
    assert encoding.b_frames == FILTER_MODULE.output_b_frames(
        filter_params, filter_args, encoding.gop
    )


def test_ffmpeg_mirror_command_contains_filter_and_random_access_options() -> None:
    config = MODULE.VideoEncodeConfig(video_codec="h264")
    params = make_video_params()
    task = MODULE.VideoTask(
        source_path=Path("/source.mp4"),
        target_path=Path("/target.mp4"),
        source_key="observation.images.top_head",
        target_key="observation.images.top_head",
        episode_index=0,
        expected_frames=10,
        params=params,
        encoding=MODULE.resolve_video_encoding(params, config),
    )

    command = MODULE.build_ffmpeg_mirror_command(task, Path("/tmp/output.mp4"), config)

    assert command[command.index("-vf") + 1] == "hflip,setpts=N/FRAME_RATE/TB"
    assert command[command.index("-map") + 1] == "0:v:0"
    assert "-an" in command
    assert command[command.index("-g") + 1] == "2"
    assert command[command.index("-bf") + 1] == "0"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-movflags") + 1] == "+faststart"


def test_probe_video_frames_uses_metadata_before_exact_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"placeholder")
    commands = []

    def fake_run(
        command: list[str],
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check
        assert capture_output
        assert text
        commands.append(command)
        if "stream=nb_frames" in command:
            return subprocess.CompletedProcess(command, 0, stdout="4\n")
        raise AssertionError("不应在 nb_frames 可用时调用精确 count_frames")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)

    assert MODULE.probe_video_frames(source_path) == 4
    assert len(commands) == 1


def test_build_video_tasks_maps_single_left_camera_to_right(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    template = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4"
    )
    source_path = source_root / template.format(
        episode_chunk=0,
        video_key="observation.images.hand_left",
        episode_index=0,
    )
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"source")
    info = {
        "chunks_size": 1000,
        "video_path": template,
        "features": {
            "observation.images.hand_left": {
                "dtype": "video",
                "info": {"video.codec": "av1"},
            }
        },
    }
    params = make_video_params()
    monkeypatch.setattr(MODULE, "ensure_video_tools", lambda: None)
    monkeypatch.setattr(MODULE, "probe_video_frames", lambda _path, exact=False: 4)
    monkeypatch.setattr(MODULE, "probe_video_params", lambda _path: params)
    monkeypatch.setattr(MODULE, "ffmpeg_has_encoder", lambda _encoder: True)

    tasks = MODULE.build_video_tasks(
        source_root,
        target_root,
        info,
        [{"episode_index": 0, "length": 4}],
        MODULE.VideoEncodeConfig(),
    )

    assert len(tasks) == 1
    assert tasks[0].source_key == "observation.images.hand_left"
    assert tasks[0].target_key == "observation.images.hand_right"
    assert "observation.images.hand_right" in str(tasks[0].target_path)

    output_info = MODULE.build_output_info(info, MODULE.VideoEncodeConfig())
    assert "observation.images.hand_left" not in output_info["features"]
    assert output_info["features"]["observation.images.hand_right"]["info"][
        "video.codec"
    ] == "h264"

    source_codec_info = MODULE.build_output_info(
        info, MODULE.VideoEncodeConfig(video_codec="source")
    )
    assert source_codec_info["features"]["observation.images.hand_right"]["info"][
        "video.codec"
    ] == "av1"


def test_build_video_tasks_maps_cam_wrist_camera_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    template = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4"
    )
    video_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    for video_key in video_keys:
        source_path = source_root / template.format(
            episode_chunk=0,
            video_key=video_key,
            episode_index=0,
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(video_key.encode())
    info = {
        "chunks_size": 1000,
        "video_path": template,
        "features": {
            video_key: {
                "dtype": "video",
                "info": {"video.codec": "av1"},
            }
            for video_key in video_keys
        },
    }
    params = make_video_params()
    monkeypatch.setattr(MODULE, "ensure_video_tools", lambda: None)
    monkeypatch.setattr(MODULE, "probe_video_frames", lambda _path, exact=False: 4)
    monkeypatch.setattr(MODULE, "probe_video_params", lambda _path: params)
    monkeypatch.setattr(MODULE, "ffmpeg_has_encoder", lambda _encoder: True)

    camera_map = MODULE.resolve_video_camera_map(info)
    tasks = MODULE.build_video_tasks(
        source_root,
        target_root,
        info,
        [{"episode_index": 0, "length": 4}],
        MODULE.VideoEncodeConfig(),
        video_camera_map=camera_map,
    )

    assert camera_map == (
        ("observation.images.cam_high", "observation.images.cam_high"),
        ("observation.images.cam_right_wrist", "observation.images.cam_left_wrist"),
        ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"),
    )
    assert len(tasks) == 3
    assert {
        task.source_key: task.target_key
        for task in tasks
    } == {
        "observation.images.cam_high": "observation.images.cam_high",
        "observation.images.cam_left_wrist": "observation.images.cam_right_wrist",
        "observation.images.cam_right_wrist": "observation.images.cam_left_wrist",
    }
    assert any(
        "observation.images.cam_left_wrist" in str(task.target_path)
        and task.source_key == "observation.images.cam_right_wrist"
        for task in tasks
    )

    output_info = MODULE.build_output_info(
        info,
        MODULE.VideoEncodeConfig(),
        camera_map,
    )
    assert set(output_info["features"]) == set(video_keys)
    assert output_info["features"]["observation.images.cam_left_wrist"]["info"][
        "video.codec"
    ] == "h264"


def test_unknown_video_camera_key_is_rejected() -> None:
    info = {
        "features": {
            "observation.images.side": {"dtype": "video"},
        },
    }

    with pytest.raises(ValueError, match="没有可用的相机镜像映射"):
        MODULE.resolve_video_camera_map(info)


def test_build_video_tasks_rejects_missing_source_before_probing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = {
        "chunks_size": 1000,
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.top_head": {"dtype": "video"},
        },
    }
    probe_called = False

    def unexpected_probe(_path: Path) -> int:
        nonlocal probe_called
        probe_called = True
        return 4

    monkeypatch.setattr(MODULE, "probe_video_frames", unexpected_probe)

    with pytest.raises(FileNotFoundError, match="发现缺失视频"):
        MODULE.build_video_tasks(
            tmp_path / "source",
            tmp_path / "target",
            info,
            [{"episode_index": 0, "length": 4}],
            MODULE.VideoEncodeConfig(),
        )

    assert not probe_called


def test_mirror_video_task_replaces_target_only_after_frame_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.mp4"
    target_path = tmp_path / "target.mp4"
    source_path.write_bytes(b"source")
    target_path.write_bytes(b"old-target")
    config = MODULE.VideoEncodeConfig(video_codec="h264")
    params = make_video_params()
    task = MODULE.VideoTask(
        source_path=source_path,
        target_path=target_path,
        source_key="observation.images.top_head",
        target_key="observation.images.top_head",
        episode_index=0,
        expected_frames=4,
        params=params,
        encoding=MODULE.resolve_video_encoding(params, config),
    )

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert check
        Path(command[-1]).write_bytes(b"new-target")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(MODULE, "probe_video_frames", lambda _path, exact=False: 4)

    MODULE.mirror_video_task(task, config)

    assert target_path.read_bytes() == b"new-target"
    assert list(tmp_path.glob("*.mirror_*.mp4")) == []


def test_mirror_video_task_preserves_target_when_frame_count_is_wrong(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.mp4"
    target_path = tmp_path / "target.mp4"
    source_path.write_bytes(b"source")
    target_path.write_bytes(b"old-target")
    config = MODULE.VideoEncodeConfig(video_codec="h264")
    params = make_video_params()
    task = MODULE.VideoTask(
        source_path=source_path,
        target_path=target_path,
        source_key="observation.images.top_head",
        target_key="observation.images.top_head",
        episode_index=0,
        expected_frames=4,
        params=params,
        encoding=MODULE.resolve_video_encoding(params, config),
    )

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert check
        Path(command[-1]).write_bytes(b"invalid-target")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(MODULE, "probe_video_frames", lambda _path, exact=False: 3)

    with pytest.raises(RuntimeError, match="视频镜像后帧数不匹配"):
        MODULE.mirror_video_task(task, config)

    assert target_path.read_bytes() == b"old-target"
    assert list(tmp_path.glob("*.mirror_*.mp4")) == []


def test_ffmpeg_cpu_codec_generates_horizontally_flipped_video(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe 不可用")
    if not MODULE.ffmpeg_has_encoder("libx264"):
        pytest.skip("libx264 编码器不可用")

    source_path = tmp_path / "source.mp4"
    target_path = tmp_path / "target.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=5",
            "-frames:v",
            "4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source_path),
        ],
        check=True,
    )
    config = MODULE.VideoEncodeConfig(video_codec="h264")
    params = MODULE.probe_video_params(source_path)
    task = MODULE.VideoTask(
        source_path=source_path,
        target_path=target_path,
        source_key="observation.images.top_head",
        target_key="observation.images.top_head",
        episode_index=0,
        expected_frames=4,
        params=params,
        encoding=MODULE.resolve_video_encoding(params, config),
    )

    MODULE.mirror_video_task(task, config)

    assert MODULE.probe_video_frames(target_path, exact=True) == 4

    def first_rgb_frame(path: Path) -> np.ndarray:
        result = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape(240, 320, 3)

    source_frame = first_rgb_frame(source_path).astype(np.int16)
    target_frame = first_rgb_frame(target_path).astype(np.int16)
    mean_absolute_error = np.abs(target_frame - source_frame[:, ::-1]).mean()
    assert mean_absolute_error < 15


def test_negate_joints_is_required_only_for_transform_commands() -> None:
    parser = MODULE.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["create-mirror", "--src-path", "/source", "--tgt-path", "/mirror"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "full",
                "--src-path",
                "/source",
                "--mirror-path",
                "/mirror",
                "--merge-path",
                "/merged",
                "--repo-id",
                "test",
            ]
        )

    args = parser.parse_args(
        [
            "merge",
            "--src-paths",
            "/source",
            "/mirror",
            "--tgt-path",
            "/merged",
            "--repo-id",
            "test",
        ]
    )
    assert args.command == "merge"


def test_video_cli_supports_hyphen_and_underscore_aliases() -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            "create-mirror",
            "--src-path",
            "/source",
            "--tgt-path",
            "/mirror",
            "--negate-joints",
            "1",
            "4",
            "6",
            "--video_codec",
            "source",
            "--b_frames",
            "1",
            "--ffmpeg_loglevel",
            "warning",
        ]
    )

    config = MODULE.video_encode_config_from_args(args)
    assert config.video_codec == "source"
    assert config.b_frames == 1
    assert config.ffmpeg_loglevel == "warning"
