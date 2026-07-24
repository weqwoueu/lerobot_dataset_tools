from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "14_kai0_time_scaling.py"
SPEC = importlib.util.spec_from_file_location("kai0_time_scaling", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

FILTER_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "11_filter_nonidle_frames.py"
FILTER_SPEC = importlib.util.spec_from_file_location(
    "filter_nonidle_frames_for_time_scaling_test", FILTER_SCRIPT_PATH
)
assert FILTER_SPEC is not None and FILTER_SPEC.loader is not None
FILTER_MODULE = importlib.util.module_from_spec(FILTER_SPEC)
sys.modules[FILTER_SPEC.name] = FILTER_MODULE
FILTER_SPEC.loader.exec_module(FILTER_MODULE)


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


def test_ffmpeg_time_scaling_command_contains_select_and_faststart() -> None:
    config = MODULE.VideoEncodeConfig(video_codec="h264")
    params = make_video_params()
    task = MODULE.VideoScaleTask(
        label="camera episode 0",
        src_path=Path("/source.mp4"),
        tgt_path=Path("/target.mp4"),
        extraction_factor=3,
        expected_source_frames=10,
        expected_frames=4,
        params=params,
        encoding=MODULE.resolve_video_encoding(params, config),
    )

    command = MODULE.build_ffmpeg_scale_command(task, Path("/tmp/output.mp4"), config)

    assert command[command.index("-vf") + 1] == "select='not(mod(n\\,3))',setpts=N/FRAME_RATE/TB"
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


def test_output_features_updates_video_codec_only_when_reencoding() -> None:
    features = {
        "observation.state": {"dtype": "float32"},
        "observation.images.cam_high": {
            "dtype": "video",
            "info": {"video.codec": "av1"},
        },
    }

    output = MODULE.output_features_for_video_config(
        features,
        MODULE.VideoEncodeConfig(video_codec="h264_nvenc"),
    )
    source = MODULE.output_features_for_video_config(
        features,
        MODULE.VideoEncodeConfig(video_codec="source"),
    )

    assert output["observation.images.cam_high"]["info"]["video.codec"] == "h264"
    assert source["observation.images.cam_high"]["info"]["video.codec"] == "av1"


def test_run_video_scale_task_replaces_target_only_after_frame_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.mp4"
    target_path = tmp_path / "target.mp4"
    source_path.write_bytes(b"source")
    target_path.write_bytes(b"old-target")
    config = MODULE.VideoEncodeConfig(video_codec="h264")
    params = make_video_params()
    task = MODULE.VideoScaleTask(
        label="camera episode 0",
        src_path=source_path,
        tgt_path=target_path,
        extraction_factor=2,
        expected_source_frames=8,
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

    MODULE.run_video_scale_task(task, config)

    assert target_path.read_bytes() == b"new-target"
    assert list(tmp_path.glob("*.time_scaling_*.mp4")) == []


def test_run_video_scale_task_preserves_target_when_frame_count_is_wrong(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.mp4"
    target_path = tmp_path / "target.mp4"
    source_path.write_bytes(b"source")
    target_path.write_bytes(b"old-target")
    config = MODULE.VideoEncodeConfig(video_codec="h264")
    params = make_video_params()
    task = MODULE.VideoScaleTask(
        label="camera episode 0",
        src_path=source_path,
        tgt_path=target_path,
        extraction_factor=2,
        expected_source_frames=8,
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

    with pytest.raises(RuntimeError, match="视频时间缩放后帧数不匹配"):
        MODULE.run_video_scale_task(task, config)

    assert target_path.read_bytes() == b"old-target"
    assert list(tmp_path.glob("*.time_scaling_*.mp4")) == []
