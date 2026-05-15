#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoFormat:
    path: Path
    codec_name: str | None
    codec_tag_string: str | None
    profile: str | None
    pix_fmt: str | None
    width: int | None
    height: int | None
    has_b_frames: int | None
    level: int | None
    refs: int | None
    bit_rate: int | None
    nb_frames: int | None
    nb_read_frames: int | None
    r_frame_rate: str | None
    avg_frame_rate: str | None
    format_name: str | None
    keyframes: int | None
    gop: int | None
    max_keyframe_gap_s: float | None
    mean_keyframe_gap_s: float | None


def _to_int(value: Any) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_video_rate(rate: str | None, default: float = 30.0) -> float:
    if not rate or rate in {"0/0", "N/A"}:
        return default
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return default


def format_fps_for_ffmpeg(fps: float) -> str:
    if abs(fps - round(fps)) < 1e-6:
        return str(int(round(fps)))
    return f"{fps:.6f}".rstrip("0").rstrip(".")


def probe_video_format(video_path: str | Path) -> VideoFormat:
    video_path = Path(video_path)
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found; cannot probe source video format")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,codec_tag_string,profile,pix_fmt,width,height,has_b_frames,level,refs,bit_rate,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate",
        "-show_entries",
        "format=format_name",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream found: {video_path}")
    stream = streams[0]
    fmt = data.get("format") or {}
    fps = parse_video_rate(stream.get("avg_frame_rate"), parse_video_rate(stream.get("r_frame_rate"), 30.0))
    keyframe_timestamps = probe_keyframe_timestamps(video_path)
    gop, max_gap_s, mean_gap_s = estimate_gop_from_keyframes(
        keyframe_timestamps,
        fps=fps,
        frame_count=_to_int(stream.get("nb_read_frames")) or _to_int(stream.get("nb_frames")),
    )
    return VideoFormat(
        path=video_path,
        codec_name=stream.get("codec_name"),
        codec_tag_string=stream.get("codec_tag_string"),
        profile=stream.get("profile"),
        pix_fmt=stream.get("pix_fmt"),
        width=_to_int(stream.get("width")),
        height=_to_int(stream.get("height")),
        has_b_frames=_to_int(stream.get("has_b_frames")),
        level=_to_int(stream.get("level")),
        refs=_to_int(stream.get("refs")),
        bit_rate=_to_int(stream.get("bit_rate")),
        nb_frames=_to_int(stream.get("nb_frames")),
        nb_read_frames=_to_int(stream.get("nb_read_frames")),
        r_frame_rate=stream.get("r_frame_rate"),
        avg_frame_rate=stream.get("avg_frame_rate"),
        format_name=fmt.get("format_name"),
        keyframes=len(keyframe_timestamps),
        gop=gop,
        max_keyframe_gap_s=max_gap_s,
        mean_keyframe_gap_s=mean_gap_s,
    )


def probe_keyframe_timestamps(video_path: str | Path) -> list[float]:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found; cannot probe source video GOP")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    timestamps = []
    for line in result.stdout.splitlines():
        value = line.strip().split(",", 1)[0]
        if value and value != "N/A":
            timestamps.append(float(value))
    return timestamps


def estimate_gop_from_keyframes(
    keyframe_timestamps: list[float],
    fps: float,
    frame_count: int | None,
) -> tuple[int | None, float | None, float | None]:
    if len(keyframe_timestamps) >= 2:
        gaps_s = [b - a for a, b in zip(keyframe_timestamps, keyframe_timestamps[1:])]
        max_gap_s = max(gaps_s)
        mean_gap_s = sum(gaps_s) / len(gaps_s)
        gop = max(1, int(round(max_gap_s * fps)))
        return gop, max_gap_s, mean_gap_s

    # Only one keyframe usually means "one intra frame at the beginning".
    # Matching that needs a keyint at least as large as the video length.
    if len(keyframe_timestamps) == 1 and frame_count:
        return max(1, frame_count), None, None

    return None, None, None


def video_fps(source_format: VideoFormat, default: float = 30.0) -> float:
    return parse_video_rate(source_format.avg_frame_rate, parse_video_rate(source_format.r_frame_rate, default))


def describe_video_format(source_format: VideoFormat) -> str:
    keyframe_info = "keyframes=?"
    if source_format.keyframes is not None:
        keyframe_info = f"keyframes={source_format.keyframes}"
        if source_format.gop is not None:
            keyframe_info += f", gop~{source_format.gop}"
        if source_format.max_keyframe_gap_s is not None:
            keyframe_info += f", max_gap={source_format.max_keyframe_gap_s:.3f}s"

    return (
        f"codec={source_format.codec_name or '?'}, "
        f"tag={source_format.codec_tag_string or '?'}, "
        f"profile={source_format.profile or '?'}, "
        f"pix_fmt={source_format.pix_fmt or '?'}, "
        f"size={source_format.width or '?'}x{source_format.height or '?'}, "
        f"fps={source_format.avg_frame_rate or source_format.r_frame_rate or '?'}, "
        f"frames={source_format.nb_read_frames or source_format.nb_frames or '?'}, "
        f"has_b_frames={source_format.has_b_frames if source_format.has_b_frames is not None else '?'}, "
        f"level={source_format.level if source_format.level is not None else '?'}, "
        f"refs={source_format.refs if source_format.refs is not None else '?'}, "
        f"bit_rate={source_format.bit_rate if source_format.bit_rate is not None else '?'}, "
        f"{keyframe_info}, "
        f"container={source_format.format_name or '?'}"
    )


def ffmpeg_encoder_for_source(source_format: VideoFormat) -> str:
    codec = (source_format.codec_name or "").lower()
    if codec == "h264":
        return "libx264"
    if codec in {"hevc", "h265"}:
        return "libx265"
    if codec == "av1":
        return "libaom-av1"
    if codec == "vp9":
        return "libvpx-vp9"
    if codec == "mpeg4":
        return "mpeg4"
    return codec or "libx264"


def opencv_fourcc_for_source(source_format: VideoFormat) -> str:
    tag = source_format.codec_tag_string
    if tag and len(tag) == 4 and not tag.startswith("["):
        return tag

    codec = (source_format.codec_name or "").lower()
    if codec == "h264":
        return "avc1"
    if codec == "mpeg4":
        return "mp4v"
    if codec == "av1":
        return "av01"
    return "mp4v"


def ffmpeg_profile_args_for_source(source_format: VideoFormat) -> list[str]:
    codec = (source_format.codec_name or "").lower()
    profile = (source_format.profile or "").lower()
    if not profile:
        return []

    if codec == "h264":
        if "baseline" in profile:
            return ["-profile:v", "baseline"]
        if "main" in profile:
            return ["-profile:v", "main"]
        if "high" in profile:
            return ["-profile:v", "high"]

    if codec in {"hevc", "h265"}:
        if "main 10" in profile:
            return ["-profile:v", "main10"]
        if "main" in profile:
            return ["-profile:v", "main"]

    return []


def ffmpeg_codec_detail_args_for_source(source_format: VideoFormat) -> list[str]:
    codec = (source_format.codec_name or "").lower()
    args: list[str] = []

    if source_format.level is not None and codec in {"h264", "hevc", "h265"}:
        args.extend(["-level:v", f"{source_format.level / 10:g}"])

    if source_format.refs is not None and codec == "h264":
        args.extend(["-refs", str(max(1, source_format.refs))])

    return args


def ffmpeg_gop_args_for_source(source_format: VideoFormat) -> list[str]:
    args: list[str] = []
    if source_format.gop is not None and source_format.gop > 0:
        args.extend([
            "-g",
            str(source_format.gop),
            "-keyint_min",
            str(source_format.gop),
            "-sc_threshold",
            "0",
        ])

    if source_format.has_b_frames is not None:
        args.extend(["-bf", str(max(0, source_format.has_b_frames))])

    return args


def ffmpeg_rate_control_args_for_source(source_format: VideoFormat) -> list[str]:
    if source_format.bit_rate is None or source_format.bit_rate <= 0:
        return []
    return ["-b:v", str(source_format.bit_rate)]


class FfmpegRawVideoWriter:
    def __init__(
        self,
        output_path: str | Path,
        source_format: VideoFormat,
        fps: float,
        width: int,
        height: int,
        ffmpeg_loglevel: str = "error",
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found; cannot encode video with source format")

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self.output_path.stem}.",
            suffix=self.output_path.suffix or ".mp4",
            dir=self.output_path.parent,
        )
        os.close(fd)
        self.tmp_path = Path(tmp_name)
        self.tmp_path.unlink(missing_ok=True)

        fps_arg = format_fps_for_ffmpeg(fps)
        output_pix_fmt = source_format.pix_fmt or "yuv420p"
        encoder = ffmpeg_encoder_for_source(source_format)
        self.cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            ffmpeg_loglevel,
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            fps_arg,
            "-i",
            "-",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            encoder,
            *ffmpeg_profile_args_for_source(source_format),
            *ffmpeg_codec_detail_args_for_source(source_format),
            *ffmpeg_gop_args_for_source(source_format),
            *ffmpeg_rate_control_args_for_source(source_format),
            "-pix_fmt",
            output_pix_fmt,
            "-r",
            fps_arg,
            "-movflags",
            "+faststart",
            str(self.tmp_path),
        ]
        self.process = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write_bgr(self, frame: Any) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        stdout, stderr = self.process.communicate()
        if self.process.returncode != 0:
            self.tmp_path.unlink(missing_ok=True)
            err = (stderr or stdout or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed with source-format encoder: {err[-2000:]}")
        self.tmp_path.replace(self.output_path)

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.communicate()
        self.tmp_path.unlink(missing_ok=True)

    def __enter__(self) -> "FfmpegRawVideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        self.close()
        return False
