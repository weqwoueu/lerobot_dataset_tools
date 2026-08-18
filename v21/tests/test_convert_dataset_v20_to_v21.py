from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "17_convert_dataset_v20_to_v21.py"
SPEC = importlib.util.spec_from_file_location("convert_dataset_v20_to_v21", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_args_defaults_to_h264_nvenc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--dataset_dir", "/tmp/dataset"])

    args = MODULE.parse_args()

    assert args.vcodec == "h264_nvenc"


@pytest.mark.parametrize(
    ("vcodec", "width", "height", "expected"),
    [
        ("h264_nvenc", 640, 480, "h264_nvenc"),
        ("hevc_nvenc", 640, 480, "hevc_nvenc"),
        ("h264_nvenc", 128, 128, "h264"),
        ("hevc_nvenc", 128, 128, "hevc"),
    ],
)
def test_resolve_output_vcodec(
    vcodec: str,
    width: int,
    height: int,
    expected: str,
) -> None:
    assert MODULE.resolve_output_vcodec(vcodec, width, height) == expected


def test_build_nvenc_ffmpeg_command_uses_training_friendly_options() -> None:
    command = MODULE.build_nvenc_ffmpeg_command(
        Path("/tmp/frames"),
        Path("/tmp/video.mp4"),
        30,
        "h264_nvenc",
    )

    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-preset") + 1] == "p4"
    assert command[command.index("-cq") + 1] == "23"
    assert command[command.index("-g") + 1] == "2"
    assert command[command.index("-bf") + 1] == "0"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-movflags") + 1] == "+faststart"
