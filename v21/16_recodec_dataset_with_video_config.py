#!/usr/bin/env python3
"""按视频参数配置重编码 LeRobot 数据集视频。

输入一个 LeRobot 数据集路径和 15_export_video_format_config.py 导出的 JSON，
脚本会复制出一个新的 `<dataset>_recodec` 数据集，并按配置重编码 videos/ 下
的 mp4。源数据集不会被修改。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取视频参数 JSON，非原地重编码 LeRobot 数据集 videos/。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        "--dataset_dir",
        type=Path,
        required=True,
        help="输入 LeRobot 数据集目录。",
    )
    parser.add_argument(
        "--config-json",
        "--config_json",
        type=Path,
        required=True,
        help="15_export_video_format_config.py 导出的视频参数配置 JSON。",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=None,
        help="输出数据集目录；默认是输入目录同级的 <name>_recodec。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行 ffmpeg 进程数。",
    )
    parser.add_argument(
        "--max-videos",
        "--max_videos",
        type=int,
        default=None,
        help="最多重编码多少个视频；调试用，未指定则重编码全部。",
    )
    parser.add_argument(
        "--ffmpeg-loglevel",
        "--ffmpeg_loglevel",
        default="error",
        help="ffmpeg 日志级别。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="如果输出目录已存在，先删除后重新生成。",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="只打印计划，不复制或重编码。",
    )
    return parser.parse_args()


def ensure_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"缺少依赖工具: {', '.join(missing)}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_dataset_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"输入数据集目录不存在或不是目录: {dataset_dir}")
    if not (dataset_dir / "videos").is_dir():
        raise FileNotFoundError(f"输入数据集 videos/ 目录不存在: {dataset_dir / 'videos'}")
    return dataset_dir


def resolve_output_dir(dataset_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return dataset_dir.with_name(f"{dataset_dir.name}_recodec")
    return output_dir.expanduser().resolve()


def iter_video_files(videos_dir: Path) -> list[Path]:
    return sorted(videos_dir.rglob("*.mp4"))


def probe_video_frame_count(video_path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if not value or value.upper() == "N/A":
        raise RuntimeError(f"无法读取视频帧数: {video_path}")
    return int(value)


def extract_video_key(video_path: Path, videos_dir: Path) -> str:
    rel_parts = video_path.relative_to(videos_dir).parts
    if len(rel_parts) >= 3 and rel_parts[0].startswith("chunk-"):
        return rel_parts[1]
    if len(rel_parts) >= 2:
        return rel_parts[-2]
    raise RuntimeError(f"无法从路径推断 video key: {video_path}")


def config_for_video_key(config: dict[str, Any], video_key: str) -> dict[str, Any]:
    video_keys = config.get("video_keys") or {}
    key_config = video_keys.get(video_key)
    if key_config is None:
        raise KeyError(f"配置文件中缺少 video key: {video_key}")
    params = key_config.get("encoding_params")
    if not params:
        raise KeyError(f"配置文件中缺少 encoding_params: {video_key}")
    return params


def sanitize_output_args(args: list[Any]) -> list[str]:
    sanitized = [str(item) for item in args]
    banned = {"-i", "-y", "-map", "-an"}
    for item in sanitized:
        if item in banned:
            raise ValueError(f"配置中的 ffmpeg_output_args 不应包含输入/全局参数: {item}")
    return sanitized


def output_args_from_params(params: dict[str, Any]) -> list[str]:
    if params.get("ffmpeg_output_args"):
        return sanitize_output_args(params["ffmpeg_output_args"])

    args = ["-c:v", str(params.get("ffmpeg_encoder") or params.get("codec_name") or "libx264")]
    profile = params.get("profile")
    codec_name = (params.get("codec_name") or "").lower()
    if codec_name == "h264" and profile:
        profile_lower = str(profile).lower()
        if "baseline" in profile_lower:
            args.extend(["-profile:v", "baseline"])
        elif "main" in profile_lower:
            args.extend(["-profile:v", "main"])
        elif "high" in profile_lower:
            args.extend(["-profile:v", "high"])

    if params.get("level") is not None and codec_name in {"h264", "hevc", "h265"}:
        args.extend(["-level:v", f"{float(params['level']) / 10:g}"])
    if params.get("refs") is not None and codec_name == "h264":
        args.extend(["-refs", str(max(1, int(params["refs"])))])
    if params.get("gop") is not None:
        gop = str(int(params["gop"]))
        args.extend(["-g", gop, "-keyint_min", gop, "-sc_threshold", str(params.get("sc_threshold", 0))])
    if params.get("bf") is not None:
        args.extend(["-bf", str(max(0, int(params["bf"])))])
    if params.get("bit_rate") is not None:
        args.extend(["-b:v", str(int(params["bit_rate"]))])

    args.extend(["-pix_fmt", str(params.get("pix_fmt") or "yuv420p")])
    fps_rate = params.get("fps_rate") or params.get("fps")
    if fps_rate is not None:
        args.extend(["-r", str(fps_rate)])
    if params.get("movflags"):
        args.extend(["-movflags", str(params["movflags"])])
    return args


def build_ffmpeg_cmd(
    src_video_path: Path,
    dst_video_path: Path,
    output_args: list[str],
    ffmpeg_loglevel: str,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        ffmpeg_loglevel,
        "-i",
        str(src_video_path),
        "-map",
        "0:v:0",
        "-an",
        *output_args,
        str(dst_video_path),
    ]


def recode_one_video(
    src_video_path: Path,
    dst_video_path: Path,
    output_args: list[str],
    ffmpeg_loglevel: str,
) -> None:
    src_frames = probe_video_frame_count(src_video_path)
    dst_video_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f"{dst_video_path.stem}.recodec_",
        suffix=dst_video_path.suffix or ".mp4",
        dir=dst_video_path.parent,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        tmp_path.unlink()
        cmd = build_ffmpeg_cmd(src_video_path, tmp_path, output_args, ffmpeg_loglevel)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        dst_frames = probe_video_frame_count(tmp_path)
        if dst_frames != src_frames:
            raise RuntimeError(f"重编码后帧数不一致: src={src_frames}, dst={dst_frames}")
        tmp_path.replace(dst_video_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def prepare_output_dataset(dataset_dir: Path, output_dir: Path, force: bool, dry_run: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"输出目录已存在，请换一个路径或使用 --force: {output_dir}")
        logger.info("删除已存在输出目录: %s", output_dir)
        if not dry_run:
            shutil.rmtree(output_dir)

    logger.info("复制数据集: %s -> %s", dataset_dir, output_dir)
    if not dry_run:
        shutil.copytree(dataset_dir, output_dir)


def metadata_codec_name(params: dict[str, Any]) -> str | None:
    codec_name = params.get("codec_name")
    codec_tag = params.get("codec_tag_string")
    if codec_name == "mpeg4" and codec_tag == "mp4v":
        return "mp4v"
    return codec_name


def update_output_info_json(output_dir: Path, config: dict[str, Any]) -> None:
    info_path = output_dir / "meta" / "info.json"
    if not info_path.is_file():
        logger.warning("输出数据集缺少 meta/info.json，跳过元数据 codec 更新: %s", info_path)
        return

    info = read_json(info_path)
    features = info.get("features", {})
    changed = False

    for video_key, key_config in (config.get("video_keys") or {}).items():
        params = key_config.get("encoding_params") or {}
        feature = features.get(video_key)
        if not feature or feature.get("dtype") != "video":
            continue

        video_info = feature.setdefault("info", {})
        codec_name = metadata_codec_name(params)
        if codec_name is not None:
            video_info["video.codec"] = codec_name
        if params.get("pix_fmt") is not None:
            video_info["video.pix_fmt"] = params["pix_fmt"]
        if params.get("height") is not None:
            video_info["video.height"] = int(params["height"])
            feature["shape"][0] = int(params["height"])
        if params.get("width") is not None:
            video_info["video.width"] = int(params["width"])
            feature["shape"][1] = int(params["width"])
        if params.get("fps") is not None:
            video_info["video.fps"] = int(round(float(params["fps"])))
        changed = True

    if changed:
        write_json(info_path, info)
        logger.info("已更新输出数据集视频元数据: %s", info_path)


def build_recode_tasks(
    dataset_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    max_videos: int | None,
) -> list[dict[str, Any]]:
    src_videos_dir = dataset_dir / "videos"
    dst_videos_dir = output_dir / "videos"
    tasks = []
    video_paths = iter_video_files(src_videos_dir)
    if max_videos is not None and max_videos > 0:
        video_paths = video_paths[:max_videos]

    for src_video_path in video_paths:
        video_key = extract_video_key(src_video_path, src_videos_dir)
        params = config_for_video_key(config, video_key)
        rel_path = src_video_path.relative_to(src_videos_dir)
        tasks.append(
            {
                "src_video_path": src_video_path,
                "dst_video_path": dst_videos_dir / rel_path,
                "video_key": video_key,
                "output_args": output_args_from_params(params),
            }
        )
    return tasks


def run_recode_tasks(tasks: list[dict[str, Any]], workers: int, ffmpeg_loglevel: str) -> None:
    if workers <= 0:
        raise ValueError(f"--workers 必须大于 0，当前为 {workers}")
    if not tasks:
        raise RuntimeError("未找到需要重编码的视频")

    logger.info("开始重编码 videos: files=%d, workers=%d", len(tasks), workers)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(
                recode_one_video,
                task["src_video_path"],
                task["dst_video_path"],
                task["output_args"],
                ffmpeg_loglevel,
            ): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                future.result()
            except Exception as e:
                rel_path = task["src_video_path"]
                raise RuntimeError(f"重编码失败: {rel_path}: {e}") from e
            completed += 1
            if completed % 20 == 0 or completed == len(tasks):
                logger.info("  已重编码 %d/%d", completed, len(tasks))


def print_plan(dataset_dir: Path, output_dir: Path, config: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    print("=" * 80)
    print("LeRobot 数据集视频重编码计划")
    print("=" * 80)
    print(f"dataset_dir : {dataset_dir}")
    print(f"output_dir  : {output_dir}")
    print(f"videos      : {len(tasks)}")
    print()

    by_key: dict[str, int] = {}
    for task in tasks:
        by_key[task["video_key"]] = by_key.get(task["video_key"], 0) + 1

    for video_key, count in sorted(by_key.items()):
        params = config_for_video_key(config, video_key)
        print(f"[{video_key}] files={count}")
        print(f"  codec={params.get('codec_name')} encoder={params.get('ffmpeg_encoder')}")
        print(
            "  "
            f"pix_fmt={params.get('pix_fmt')} "
            f"fps={params.get('fps_rate') or params.get('fps')} "
            f"gop={params.get('gop')} "
            f"bf={params.get('bf')} "
            f"level={params.get('level')} "
            f"refs={params.get('refs')} "
            f"bit_rate={params.get('bit_rate')}"
        )
        print(f"  ffmpeg_output_args={' '.join(output_args_from_params(params))}")
    print()


def main() -> int:
    args = parse_args()
    ensure_tools()

    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    config_json = args.config_json.expanduser().resolve()
    if not config_json.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_json}")
    output_dir = resolve_output_dir(dataset_dir, args.output_dir)
    config = read_json(config_json)

    tasks = build_recode_tasks(dataset_dir, output_dir, config, args.max_videos)
    print_plan(dataset_dir, output_dir, config, tasks)

    if args.dry_run:
        logger.info("[dry-run] 只打印计划，不复制或重编码")
        return 0

    prepare_output_dataset(dataset_dir, output_dir, args.force, args.dry_run)
    update_output_info_json(output_dir, config)
    run_recode_tasks(tasks, args.workers, args.ffmpeg_loglevel)
    logger.info("完成: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
