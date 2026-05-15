#!/usr/bin/env python3
"""
extract_lerobot.py

Extract (downsample) frames from a LeRobot dataset by keeping every Nth frame.
This accelerates the actions in the dataset - for example, with extraction_factor=2,
a 60-frame episode becomes a 30-frame episode.

Usage:
    python extract_lerobot.py --src_path /path/to/source --tgt_path /path/to/target \\
                              --repo_id extracted_dataset --extraction_factor 2 --num-workers 4
"""
from pathlib import Path
import importlib.util
import shutil
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from video_format_utils import (
    FfmpegRawVideoWriter,
    describe_video_format,
    opencv_fourcc_for_source,
    probe_video_format,
)

# --- lerobot imports (must be available in env) ---
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import (
        load_info,
        load_episodes,
        load_episodes_stats,
        load_tasks,
    )
except Exception as e:
    raise RuntimeError("lerobot package import failed. Activate environment where lerobot is installed.") from e

# Try to import video processing libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not available. Video extraction will be skipped.")

# 优先使用 torchcodec 解码（支持 AV1 等），否则回退到 OpenCV
TORCHCODEC_AVAILABLE = importlib.util.find_spec("torchcodec") is not None
if TORCHCODEC_AVAILABLE:
    import torch
    from torchcodec.decoders import VideoDecoder

_DECODER_LIB_PRINTED = False  # 仅打印一次使用的解码库

# Try to import merge function
try:
    from merge_lerobot import merge_repos
    MERGE_AVAILABLE = True
except ImportError:
    MERGE_AVAILABLE = False
    print("Warning: merge_lerobot module not available. Merge functionality will be disabled.")


def find_parquet_by_episode(src_root: Path, ep_idx: int) -> Path | None:
    """Try to locate parquet for given episode index under src_root."""
    basename = f"episode_{ep_idx:06d}.parquet"
    candidates = list(src_root.rglob(basename))
    return candidates[0] if candidates else None


def find_video_by_episode_and_key(src_root: Path, ep_idx: int, vid_key: str) -> Path | None:
    """Try to locate mp4 for given episode and video key under src_root."""
    basename = f"episode_{ep_idx:06d}.mp4"
    candidates = list(src_root.rglob(basename))
    for c in candidates:
        if vid_key in "/".join(c.parts):
            return c
    return candidates[0] if candidates else None


def write_parquet_like_source(df: pd.DataFrame, src_parquet_path: Path, tgt_parquet_path: Path) -> None:
    """Write parquet with the same Arrow field types/order as the source episode."""
    source_schema = pq.read_schema(str(src_parquet_path))
    source_names = source_schema.names

    ordered_columns = [name for name in source_names if name in df.columns]
    extra_columns = [name for name in df.columns if name not in source_names]
    output_df = df[ordered_columns + extra_columns]

    if not extra_columns and len(ordered_columns) == len(source_names):
        target_schema = source_schema
    else:
        inferred_schema = pa.Table.from_pandas(output_df, preserve_index=False).schema
        target_schema = pa.schema([
            source_schema.field(name) if name in source_names else inferred_schema.field(name)
            for name in output_df.columns
        ])

    table = pa.Table.from_pandas(output_df, schema=target_schema, preserve_index=False)
    pq.write_table(table, str(tgt_parquet_path))


def _extract_frames_from_video_torchcodec(
    src_video_path: Path, tgt_video_path: Path, extraction_factor: int, target_fps: float
) -> int:
    """使用 torchcodec 解码并抽帧（支持 AV1 等格式）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        decoder = VideoDecoder(str(src_video_path), device=device, seek_mode="approximate")
    except RuntimeError as e:
        if "Unsupported device" in str(e):
            device = "cpu"
            decoder = VideoDecoder(str(src_video_path), device=device, seek_mode="approximate")
        else:
            raise
    source_format = probe_video_format(src_video_path)
    metadata = decoder.metadata
    width = metadata.width
    height = metadata.height
    tgt_video_path.parent.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    try:
        with FfmpegRawVideoWriter(tgt_video_path, source_format, target_fps, width, height) as out:
            for frame_idx, frame in enumerate(decoder):
                if frame_idx % extraction_factor == 0:
                    frame_np = frame.cpu().permute(1, 2, 0).numpy()
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    out.write_bgr(frame_bgr)
                    extracted_count += 1
    except Exception:
        fourcc_name = opencv_fourcc_for_source(source_format)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        out = cv2.VideoWriter(str(tgt_video_path), fourcc, target_fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Cannot create output video with source format ({fourcc_name}): {tgt_video_path}")

        decoder = VideoDecoder(str(src_video_path), device=device, seek_mode="approximate")
        extracted_count = 0
        for frame_idx, frame in enumerate(decoder):
            if frame_idx % extraction_factor == 0:
                frame_np = frame.cpu().permute(1, 2, 0).numpy()
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                extracted_count += 1
        out.release()
    return extracted_count


def _extract_frames_from_video_opencv(
    src_video_path: Path, tgt_video_path: Path, extraction_factor: int, target_fps: float
) -> int:
    """
    使用 OpenCV 解码并抽帧
    Extract every Nth frame from source video and write to target video.
    
    Args:
        src_video_path: Source video file path
        tgt_video_path: Target video file path
        extraction_factor: Keep every Nth frame
        target_fps: FPS for the output video
    """
    cap = cv2.VideoCapture(str(src_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_video_path}")
    
    source_format = probe_video_format(src_video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tgt_video_path.parent.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    extracted_count = 0

    try:
        with FfmpegRawVideoWriter(tgt_video_path, source_format, target_fps, width, height) as out:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Keep every Nth frame (0, N, 2N, 3N, ...)
                if frame_idx % extraction_factor == 0:
                    out.write_bgr(frame)
                    extracted_count += 1

                frame_idx += 1
    except Exception:
        cap.release()
        cap = cv2.VideoCapture(str(src_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot reopen video: {src_video_path}")
        fourcc_name = opencv_fourcc_for_source(source_format)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        out = cv2.VideoWriter(str(tgt_video_path), fourcc, target_fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video with source format ({fourcc_name}): {tgt_video_path}")

        frame_idx = 0
        extracted_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % extraction_factor == 0:
                out.write(frame)
                extracted_count += 1

            frame_idx += 1
        out.release()
    finally:
        cap.release()
    
    return extracted_count


def extract_frames_from_video(
    src_video_path: Path,
    tgt_video_path: Path,
    extraction_factor: int,
    target_fps: float,
    announce_decoder: bool = True,
):
    """
    Extract every Nth frame from source video and write to target video.
    优先使用 torchcodec（支持 AV1），否则使用 OpenCV。

    Args:
        src_video_path: Source video file path
        tgt_video_path: Target video file path
        extraction_factor: Keep every Nth frame
        target_fps: FPS for the output video
        announce_decoder: Print decoder backend once in the current process
    """
    if not TORCHCODEC_AVAILABLE:
        if not HAS_CV2:
            raise RuntimeError("cv2 is required for video extraction but not available")

    global _DECODER_LIB_PRINTED
    if announce_decoder and not _DECODER_LIB_PRINTED:
        _DECODER_LIB_PRINTED = True
        _lib = "torchcodec" if TORCHCODEC_AVAILABLE else "opencv"
        print(f"[*] 视频解码库: {_lib}")

    if TORCHCODEC_AVAILABLE:
        try:
            return _extract_frames_from_video_torchcodec(src_video_path, tgt_video_path, extraction_factor, target_fps)
        except Exception:
            if not HAS_CV2:
                raise RuntimeError("torchcodec failed and cv2 is not available for fallback")
            return _extract_frames_from_video_opencv(src_video_path, tgt_video_path, extraction_factor, target_fps)
    return _extract_frames_from_video_opencv(src_video_path, tgt_video_path, extraction_factor, target_fps)


def _extract_video_task(task: Dict[str, Any]) -> Tuple[str, bool, str]:
    label = task["label"]
    try:
        extracted_count = extract_frames_from_video(
            Path(task["src_path"]),
            Path(task["tgt_path"]),
            int(task["extraction_factor"]),
            float(task["fps"]),
            announce_decoder=False,
        )
        return label, True, f"Extracted {extracted_count} frames"
    except Exception as e:
        return label, False, f"Failed to extract {label}: {e}"


def _copy_video_task(task: Dict[str, Any]) -> Tuple[str, bool, str]:
    label = task["label"]
    try:
        src_path = Path(task["src_path"])
        tgt_path = Path(task["tgt_path"])
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_path), str(tgt_path))
        return label, True, "Copied video"
    except Exception as e:
        return label, False, f"Failed to copy {label}: {e}"


def _run_video_tasks(tasks: List[Dict[str, Any]], worker_fn, desc: str, num_workers: int) -> List[str]:
    if not tasks:
        return []
    if num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {num_workers}")

    print(f"[*] {desc}: {len(tasks)} files, workers={num_workers}")
    warnings: List[str] = []
    success_count = 0
    fail_count = 0

    if num_workers == 1:
        results_iter = (worker_fn(task) for task in tasks)
        for _, success, message in tqdm(results_iter, total=len(tasks), desc=desc):
            if success:
                success_count += 1
            else:
                fail_count += 1
                warnings.append(message)
                print(f"  [!] {message}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_label = {
                executor.submit(worker_fn, task): task["label"]
                for task in tasks
            }

            for future in tqdm(as_completed(future_to_label), total=len(tasks), desc=desc):
                label = future_to_label[future]
                try:
                    _, success, message = future.result()
                except Exception as e:
                    success = False
                    message = f"Failed to process {label}: {e}"

                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    warnings.append(message)
                    print(f"  [!] {message}")

    print(f"[*] {desc} complete: {success_count} succeeded, {fail_count} failed")
    return warnings


def process_video_extraction_tasks(tasks: List[Dict[str, Any]], num_workers: int) -> List[str]:
    if tasks:
        try:
            source_format = probe_video_format(tasks[0]["src_path"])
            print(f"[*] 源视频格式样例: {Path(tasks[0]['src_path']).name} -> {describe_video_format(source_format)}")
        except Exception as e:
            print(f"[!] 无法读取源视频格式样例: {tasks[0]['src_path']}: {e}")

        global _DECODER_LIB_PRINTED
        if not _DECODER_LIB_PRINTED:
            _DECODER_LIB_PRINTED = True
            _lib = "torchcodec" if TORCHCODEC_AVAILABLE else "opencv"
            print(f"[*] 视频解码库: {_lib}")
    return _run_video_tasks(tasks, _extract_video_task, "Extracting videos", num_workers)


def process_video_copy_tasks(tasks: List[Dict[str, Any]], num_workers: int) -> List[str]:
    return _run_video_tasks(tasks, _copy_video_task, "Copying videos", num_workers)


def extract_dataset(
    src_path: str,
    tgt_path: str,
    repo_id: str,
    extraction_factor: int = 2,
    num_workers: int = 4,
    force: bool = False,
    merge_src_paths: List[str] = None,
    merge_tgt_path: str = None,
    merge_repo_id: str = None,
    merge_force: bool = False,
):
    """
    Extract frames from a LeRobot dataset by keeping every Nth frame.
    
    Args:
        src_path: Path to source LeRobot dataset
        tgt_path: Path to target (extracted) dataset
        repo_id: Repository ID for the new dataset
        extraction_factor: Keep every Nth frame (e.g., 2 means keep frames 0, 2, 4, ...)
        num_workers: Number of parallel worker processes for video processing
        force: Force extraction even if target exists
        merge_src_paths: Optional list of additional source dataset paths to merge with extracted dataset
        merge_tgt_path: Optional path for merged dataset (if None and merge_src_paths provided, uses tgt_path)
        merge_repo_id: Optional repo_id for merged dataset (if None and merge_src_paths provided, uses repo_id)
        merge_force: Force merge even if conflicts exist
    """
    if extraction_factor < 1:
        raise ValueError(f"extraction_factor must be >= 1, got {extraction_factor}")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {num_workers}")
    
    src_root = Path(src_path).expanduser().resolve()
    if not src_root.exists():
        raise RuntimeError(f"Source path does not exist: {src_path}")
    
    tgt_root = Path(tgt_path).expanduser().resolve()
    if tgt_root.exists() and any(tgt_root.iterdir()) and not force:
        raise RuntimeError(f"Target {tgt_root} exists and is not empty. Use --force or remove it first.")
    
    # Load source dataset info
    print(f"[*] Loading source dataset from {src_root}")
    try:
        src_info = load_info(src_root)
        src_episodes = load_episodes(src_root)
        src_tasks_dict, _ = load_tasks(src_root)
    except Exception as e:
        raise RuntimeError(f"Cannot load source dataset: {e}")
    
    try:
        src_episodes_stats = load_episodes_stats(src_root)
    except Exception:
        src_episodes_stats = {}
    
    # Extract parameters from source
    fps = src_info.get("fps", 30)
    robot_type = src_info.get("robot_type", "unknown")
    features = src_info.get("features", {})
    
    print(f"[*] Source dataset info:")
    print(f"    - FPS: {fps}")
    print(f"    - Robot type: {robot_type}")
    print(f"    - Total episodes: {src_info.get('total_episodes', 0)}")
    print(f"    - Total frames: {src_info.get('total_frames', 0)}")
    print(f"    - Extraction factor: {extraction_factor}")
    print(f"    - Video workers: {num_workers}")
    
    # Create target dataset structure
    print(f"[*] Creating target dataset at {tgt_root}")
    ds_target = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        root=str(tgt_root),
        robot_type=robot_type,
        features=features
    )
    meta_target = ds_target.meta
    
    # Add tasks from source
    for k, task in sorted(src_tasks_dict.items(), key=lambda kv: int(kv[0])):
        meta_target.add_task(task)
    print(f"[*] Added {len(src_tasks_dict)} tasks to target dataset")
    
    # Determine video keys
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    print(f"[*] Video keys to process: {video_keys}")
    
    warnings: List[str] = []
    video_tasks: List[Dict[str, Any]] = []
    
    # Process each episode
    items_sorted = sorted(src_episodes.items(), key=lambda kv: int(kv[0]))
    for src_ep_idx_str, ep in tqdm(items_sorted, desc="Extracting episodes"):
        src_ep_idx = int(src_ep_idx_str)
        src_ep_length = int(ep.get("length", 0))
        ep_tasks = ep.get("tasks", [])
        
        # Calculate new episode length after extraction
        new_ep_length = (src_ep_length + extraction_factor - 1) // extraction_factor
        if new_ep_length == 0:
            warnings.append(f"Episode {src_ep_idx} too short ({src_ep_length} frames), skipping")
            continue
        
        # Find source parquet file
        src_chunksize = int(src_info.get("chunks_size", 1000))
        src_chunk_idx = src_ep_idx // src_chunksize
        
        try:
            src_parquet_rel = src_info["data_path"].format(
                episode_chunk=src_chunk_idx,
                episode_index=src_ep_idx
            )
            src_parquet_path = (src_root / src_parquet_rel).resolve()
            if not src_parquet_path.is_file():
                alt = find_parquet_by_episode(src_root, src_ep_idx)
                if alt:
                    src_parquet_path = alt
                else:
                    raise FileNotFoundError
        except Exception:
            alt = find_parquet_by_episode(src_root, src_ep_idx)
            if alt:
                src_parquet_path = alt
            else:
                warnings.append(f"Parquet for episode {src_ep_idx} not found, skipping")
                continue
        
        # New episode index in target
        new_ep_idx = int(meta_target.info["total_episodes"])
        
        # Target paths
        tgt_chunk_idx = meta_target.get_episode_chunk(new_ep_idx)
        tgt_parquet_rel = meta_target.data_path.format(
            episode_chunk=tgt_chunk_idx,
            episode_index=new_ep_idx
        )
        tgt_parquet_path = meta_target.root / tgt_parquet_rel
        
        # --- EXTRACT & PATCH PARQUET: keep every Nth frame ---
        start_global_index = int(meta_target.info.get("total_frames", 0))
        tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df = pd.read_parquet(str(src_parquet_path))
            total_frames = len(df)
            
            # Extract every Nth frame (indices 0, N, 2N, 3N, ...)
            extracted_indices = list(range(0, total_frames, extraction_factor))
            df_extracted = df.iloc[extracted_indices].copy()
            
            # Reset pandas index immediately to avoid alignment issues
            df_extracted.reset_index(drop=True, inplace=True)
            
            n_rows = len(df_extracted)
            
            # Helper function to handle episode_index column format
            def make_episode_index_col(series, scalar_val, n):
                if len(series) > 0 and series.dtype == object and isinstance(series.iloc[0], (list, tuple, np.ndarray)):
                    return pd.Series([[int(scalar_val)]] * n, index=range(n))
                else:
                    return pd.Series([int(scalar_val)] * n, index=range(n))
            
            # Update episode_index
            df_extracted["episode_index"] = make_episode_index_col(df_extracted["episode_index"], new_ep_idx, n_rows)
            
            # Update frame_index to be sequential within the episode [0, 1, 2, 3, ...]
            if "frame_index" in df_extracted.columns:
                new_frame_indices = list(range(n_rows))
                if df_extracted["frame_index"].dtype == object and n_rows > 0:
                    first_val = df_extracted["frame_index"].iloc[0]
                    if isinstance(first_val, (list, tuple, np.ndarray)):
                        df_extracted["frame_index"] = [[int(x)] for x in new_frame_indices]
                    else:
                        df_extracted["frame_index"] = new_frame_indices
                else:
                    df_extracted["frame_index"] = new_frame_indices
            
            # Update timestamp to be evenly spaced according to FPS
            if "timestamp" in df_extracted.columns:
                frame_duration = 1.0 / fps
                new_timestamps = (np.arange(n_rows, dtype=np.float32) * np.float32(frame_duration)).astype(np.float32)
                if df_extracted["timestamp"].dtype == object and n_rows > 0:
                    first_val = df_extracted["timestamp"].iloc[0]
                    if isinstance(first_val, (list, tuple, np.ndarray)):
                        df_extracted["timestamp"] = [[float(x)] for x in new_timestamps]
                    else:
                        df_extracted["timestamp"] = new_timestamps
                else:
                    df_extracted["timestamp"] = new_timestamps
            
            # Update global index
            new_global_indices = list(range(start_global_index, start_global_index + n_rows))
            if "index" in df_extracted.columns:
                if df_extracted["index"].dtype == object and n_rows > 0:
                    first_val = df_extracted["index"].iloc[0]
                    if isinstance(first_val, (list, tuple, np.ndarray)):
                        df_extracted["index"] = [[int(x)] for x in new_global_indices]
                    else:
                        df_extracted["index"] = new_global_indices
                else:
                    df_extracted["index"] = new_global_indices
            else:
                    df_extracted["index"] = new_global_indices
            
            # Write extracted parquet
            write_parquet_like_source(df_extracted, src_parquet_path, tgt_parquet_path)
            
        except Exception as e:
            warnings.append(f"Failed to extract parquet for episode {src_ep_idx}: {e}")
            continue
        
        # --- EXTRACT VIDEOS ---
        for vid_key in video_keys:
            try:
                src_vpath_rel = src_info["video_path"].format(
                    episode_chunk=src_chunk_idx,
                    video_key=vid_key,
                    episode_index=src_ep_idx
                )
                src_vpath = (src_root / src_vpath_rel).resolve()
                if not src_vpath.is_file():
                    alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                    if alt_vid:
                        src_vpath = alt_vid
                    else:
                        warnings.append(f"Video {vid_key} for episode {src_ep_idx} not found, skipping")
                        continue
            except Exception:
                alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                if alt_vid:
                    src_vpath = alt_vid
                else:
                    warnings.append(f"Video {vid_key} for episode {src_ep_idx} not found, skipping")
                    continue
            
            tgt_vchunk_idx = tgt_chunk_idx
            tgt_vpath_rel = meta_target.video_path.format(
                episode_chunk=tgt_vchunk_idx,
                video_key=vid_key,
                episode_index=new_ep_idx
            )
            tgt_vpath = meta_target.root / tgt_vpath_rel
            
            video_tasks.append({
                "label": f"{vid_key} episode {src_ep_idx}",
                "src_path": str(src_vpath),
                "tgt_path": str(tgt_vpath),
                "extraction_factor": extraction_factor,
                "fps": float(fps),
            })
        
        # Get episode stats (may use string keys)
        ep_stats = {}
        if isinstance(src_episodes_stats, dict):
            ep_stats = src_episodes_stats.get(str(src_ep_idx), src_episodes_stats.get(src_ep_idx, {}))
        
        # Register episode in target meta
        meta_target.save_episode(
            episode_index=new_ep_idx,
            episode_length=new_ep_length,
            episode_tasks=ep_tasks,
            episode_stats=ep_stats
        )

    warnings.extend(process_video_extraction_tasks(video_tasks, num_workers))
    
    # Summary
    print("\n[+] Extraction finished!")
    print(f"    Target total_episodes: {meta_target.info.get('total_episodes')}")
    print(f"    Target total_frames  : {meta_target.info.get('total_frames')}")
    print(f"    Reduction factor     : ~{extraction_factor}x")
    
    if warnings:
        print(f"\n[!] {len(warnings)} warnings:")
        for w in warnings[:10]:  # Show first 10 warnings
            print("  -", w)
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    # Smoke test
    try:
        ds_check = LeRobotDataset(repo_id=repo_id, root=str(meta_target.root))
        print("\n[*] Smoke test succeeded!")
        print(f"    Loaded dataset: {ds_check.num_episodes} episodes, {ds_check.num_frames} frames")
    except Exception as e:
        print(f"\n[!] Warning: failed to load target dataset: {e}")
    
    # Merge with additional datasets if requested
    if merge_src_paths and len(merge_src_paths) > 0:
        if not MERGE_AVAILABLE:
            print("\n[!] Warning: merge_lerobot module not available. Skipping merge step.")
            return
        
        print("\n" + "=" * 60)
        print("Merging extracted dataset with additional sources")
        print("=" * 60)
        
        # Prepare merge source paths (include extracted dataset)
        all_merge_srcs = [tgt_path] + merge_src_paths
        merge_final_path = merge_tgt_path if merge_tgt_path else tgt_path + "_merged"
        merge_final_repo_id = merge_repo_id if merge_repo_id else repo_id + "_merged"
        
        print(f"Merge sources: {all_merge_srcs}")
        print(f"Merge target: {merge_final_path}")
        print(f"Merge repo_id: {merge_final_repo_id}")
        print()
        
        try:
            # Infer features/fps/robot_type from extracted dataset
            merge_repos(
                src_paths=all_merge_srcs,
                tgt_path=merge_final_path,
                repo_id=merge_final_repo_id,
                fps=fps,
                robot_type=robot_type,
                features=features,
                force=merge_force
            )
            print("\n[+] Merge complete!")
        except Exception as e:
            print(f"\n[!] Error during merge: {e}")
            raise
    
    return


def time_scaling_with_split(
    src_path: str,
    tgt_path: str,
    repo_id: str,
    split_ratio: float = 0.3,
    extraction_factor: int = 2,
    num_workers: int = 4,
    force: bool = False,
):
    """
    Split dataset by ratio, extract frames from one part, and merge both parts.
    
    Args:
        src_path: Path to source LeRobot dataset
        tgt_path: Path to target (final merged) dataset
        repo_id: Repository ID for the new dataset (will append _time_scaling suffix)
        split_ratio: Ratio of data to extract (e.g., 0.3 means 30% will be extracted, 70% kept original)
        extraction_factor: Extract every Nth frame for the extracted portion
        num_workers: Number of parallel worker processes for video processing
        force: Force operation even if target exists
    """
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")
    
    if extraction_factor < 1:
        raise ValueError(f"extraction_factor must be >= 1, got {extraction_factor}")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {num_workers}")
    
    src_root = Path(src_path).expanduser().resolve()
    if not src_root.exists():
        raise RuntimeError(f"Source path does not exist: {src_path}")
    
    tgt_root = Path(tgt_path).expanduser().resolve()
    if tgt_root.exists() and any(tgt_root.iterdir()) and not force:
        raise RuntimeError(f"Target {tgt_root} exists and is not empty. Use --force or remove it first.")
    
    # Load source dataset
    print(f"[*] Loading source dataset from {src_root}")
    try:
        src_info = load_info(src_root)
        src_episodes = load_episodes(src_root)
        src_tasks_dict, _ = load_tasks(src_root)
    except Exception as e:
        raise RuntimeError(f"Cannot load source dataset: {e}")
    
    try:
        src_episodes_stats = load_episodes_stats(src_root)
    except Exception:
        src_episodes_stats = {}
    
    # Extract parameters from source
    fps = src_info.get("fps", 30)
    robot_type = src_info.get("robot_type", "unknown")
    features = src_info.get("features", {})
    
    total_episodes = len(src_episodes)
    extract_count = max(1, int(total_episodes * split_ratio))
    keep_count = total_episodes - extract_count
    
    print(f"[*] Source dataset info:")
    print(f"    - FPS: {fps}")
    print(f"    - Robot type: {robot_type}")
    print(f"    - Total episodes: {total_episodes}")
    print(f"    - Split ratio: {split_ratio}")
    print(f"    - Episodes to extract: {extract_count}")
    print(f"    - Episodes to keep original: {keep_count}")
    print(f"    - Extraction factor: {extraction_factor}")
    print(f"    - Video workers: {num_workers}")
    
    # Sort episodes by index
    items_sorted = sorted(src_episodes.items(), key=lambda kv: int(kv[0]))
    
    # Split episodes
    extract_episodes = dict(items_sorted[:extract_count])
    keep_episodes = dict(items_sorted[extract_count:])
    
    print(f"\n[*] Split episodes:")
    print(f"    - Extracting: episodes {list(extract_episodes.keys())[0]} to {list(extract_episodes.keys())[-1]}")
    print(f"    - Keeping original: episodes {list(keep_episodes.keys())[0]} to {list(keep_episodes.keys())[-1]}")
    
    # Create temporary directories
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="time_scaling_"))
    extract_tgt_path = str(temp_dir / "extracted")
    keep_tgt_path = str(temp_dir / "kept")
    
    try:
        # Step 1: Extract frames from first portion
        print(f"\n[1/3] Extracting frames from {extract_count} episodes...")
        # Create target dataset structure for extracted portion
        ds_extract = LeRobotDataset.create(
            repo_id=repo_id + "_extracted",
            fps=int(fps),
            root=extract_tgt_path,
            robot_type=robot_type,
            features=features
        )
        meta_extract = ds_extract.meta
        
        # Add tasks
        for k, task in sorted(src_tasks_dict.items(), key=lambda kv: int(kv[0])):
            meta_extract.add_task(task)
        
        video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
        warnings: List[str] = []
        extract_video_tasks: List[Dict[str, Any]] = []
        
        # Process extract episodes
        for src_ep_idx_str, ep in tqdm(extract_episodes.items(), desc="Extracting episodes"):
            src_ep_idx = int(src_ep_idx_str)
            src_ep_length = int(ep.get("length", 0))
            ep_tasks = ep.get("tasks", [])
            
            # Calculate new episode length after extraction
            new_ep_length = (src_ep_length + extraction_factor - 1) // extraction_factor
            if new_ep_length == 0:
                warnings.append(f"Episode {src_ep_idx} too short ({src_ep_length} frames), skipping")
                continue
            
            # Find source parquet file
            src_chunksize = int(src_info.get("chunks_size", 1000))
            src_chunk_idx = src_ep_idx // src_chunksize
            
            try:
                src_parquet_rel = src_info["data_path"].format(
                    episode_chunk=src_chunk_idx,
                    episode_index=src_ep_idx
                )
                src_parquet_path = (src_root / src_parquet_rel).resolve()
                if not src_parquet_path.is_file():
                    alt = find_parquet_by_episode(src_root, src_ep_idx)
                    if alt:
                        src_parquet_path = alt
                    else:
                        raise FileNotFoundError
            except Exception:
                alt = find_parquet_by_episode(src_root, src_ep_idx)
                if alt:
                    src_parquet_path = alt
                else:
                    warnings.append(f"Parquet for episode {src_ep_idx} not found, skipping")
                    continue
            
            # New episode index in target
            new_ep_idx = int(meta_extract.info["total_episodes"])
            
            # Target paths
            tgt_chunk_idx = meta_extract.get_episode_chunk(new_ep_idx)
            tgt_parquet_rel = meta_extract.data_path.format(
                episode_chunk=tgt_chunk_idx,
                episode_index=new_ep_idx
            )
            tgt_parquet_path = meta_extract.root / tgt_parquet_rel
            
            # Extract & patch parquet
            start_global_index = int(meta_extract.info.get("total_frames", 0))
            tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                df = pd.read_parquet(str(src_parquet_path))
                total_frames = len(df)
                
                # Extract every Nth frame
                extracted_indices = list(range(0, total_frames, extraction_factor))
                df_extracted = df.iloc[extracted_indices].copy()
                df_extracted.reset_index(drop=True, inplace=True)
                
                n_rows = len(df_extracted)
                
                def make_episode_index_col(series, scalar_val, n):
                    if len(series) > 0 and series.dtype == object and isinstance(series.iloc[0], (list, tuple, np.ndarray)):
                        return pd.Series([[int(scalar_val)]] * n, index=range(n))
                    else:
                        return pd.Series([int(scalar_val)] * n, index=range(n))
                
                df_extracted["episode_index"] = make_episode_index_col(df_extracted["episode_index"], new_ep_idx, n_rows)
                
                # Update frame_index
                if "frame_index" in df_extracted.columns:
                    new_frame_indices = list(range(n_rows))
                    if df_extracted["frame_index"].dtype == object and n_rows > 0:
                        first_val = df_extracted["frame_index"].iloc[0]
                        if isinstance(first_val, (list, tuple, np.ndarray)):
                            df_extracted["frame_index"] = [[int(x)] for x in new_frame_indices]
                        else:
                            df_extracted["frame_index"] = new_frame_indices
                    else:
                        df_extracted["frame_index"] = new_frame_indices
                
                # Update timestamp
                if "timestamp" in df_extracted.columns:
                    frame_duration = 1.0 / fps
                    new_timestamps = (np.arange(n_rows, dtype=np.float32) * np.float32(frame_duration)).astype(np.float32)
                    if df_extracted["timestamp"].dtype == object and n_rows > 0:
                        first_val = df_extracted["timestamp"].iloc[0]
                        if isinstance(first_val, (list, tuple, np.ndarray)):
                            df_extracted["timestamp"] = [[float(x)] for x in new_timestamps]
                        else:
                            df_extracted["timestamp"] = new_timestamps
                    else:
                        df_extracted["timestamp"] = new_timestamps
                
                # Update global index
                new_global_indices = list(range(start_global_index, start_global_index + n_rows))
                if "index" in df_extracted.columns:
                    if df_extracted["index"].dtype == object and n_rows > 0:
                        first_val = df_extracted["index"].iloc[0]
                        if isinstance(first_val, (list, tuple, np.ndarray)):
                            df_extracted["index"] = [[int(x)] for x in new_global_indices]
                        else:
                            df_extracted["index"] = new_global_indices
                    else:
                        df_extracted["index"] = new_global_indices
                else:
                    df_extracted["index"] = new_global_indices
                
                write_parquet_like_source(df_extracted, src_parquet_path, tgt_parquet_path)
            except Exception as e:
                warnings.append(f"Failed to extract parquet for episode {src_ep_idx}: {e}")
                continue
            
            # Extract videos
            for vid_key in video_keys:
                try:
                    src_vpath_rel = src_info["video_path"].format(
                        episode_chunk=src_chunk_idx,
                        video_key=vid_key,
                        episode_index=src_ep_idx
                    )
                    src_vpath = (src_root / src_vpath_rel).resolve()
                    if not src_vpath.is_file():
                        alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                        if alt_vid:
                            src_vpath = alt_vid
                        else:
                            continue
                except Exception:
                    alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                    if alt_vid:
                        src_vpath = alt_vid
                    else:
                        continue
                
                tgt_vchunk_idx = tgt_chunk_idx
                tgt_vpath_rel = meta_extract.video_path.format(
                    episode_chunk=tgt_vchunk_idx,
                    video_key=vid_key,
                    episode_index=new_ep_idx
                )
                tgt_vpath = meta_extract.root / tgt_vpath_rel
                
                extract_video_tasks.append({
                    "label": f"{vid_key} episode {src_ep_idx}",
                    "src_path": str(src_vpath),
                    "tgt_path": str(tgt_vpath),
                    "extraction_factor": extraction_factor,
                    "fps": float(fps),
                })
            
            # Get episode stats
            ep_stats = {}
            if isinstance(src_episodes_stats, dict):
                ep_stats = src_episodes_stats.get(str(src_ep_idx), src_episodes_stats.get(src_ep_idx, {}))
            
            meta_extract.save_episode(
                episode_index=new_ep_idx,
                episode_length=new_ep_length,
                episode_tasks=ep_tasks,
                episode_stats=ep_stats
            )

        warnings.extend(process_video_extraction_tasks(extract_video_tasks, num_workers))
        
        if warnings:
            print(f"  [!] {len(warnings)} warnings during extraction")
        
        # Step 2: Copy original episodes for second portion
        print(f"\n[2/3] Copying original {keep_count} episodes...")
        # Create a new dataset with only the kept episodes
        ds_keep = LeRobotDataset.create(
            repo_id=repo_id + "_kept",
            fps=int(fps),
            root=keep_tgt_path,
            robot_type=robot_type,
            features=features
        )
        meta_keep = ds_keep.meta
        
        # Add tasks
        for k, task in sorted(src_tasks_dict.items(), key=lambda kv: int(kv[0])):
            meta_keep.add_task(task)
        
        video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
        copy_video_tasks: List[Dict[str, Any]] = []
        
        for src_ep_idx_str, ep in tqdm(keep_episodes.items(), desc="Copying episodes"):
            src_ep_idx = int(src_ep_idx_str)
            ep_length = int(ep.get("length", 0))
            ep_tasks = ep.get("tasks", [])
            
            # Find source parquet
            src_chunksize = int(src_info.get("chunks_size", 1000))
            src_chunk_idx = src_ep_idx // src_chunksize
            
            try:
                src_parquet_rel = src_info["data_path"].format(
                    episode_chunk=src_chunk_idx,
                    episode_index=src_ep_idx
                )
                src_parquet_path = (src_root / src_parquet_rel).resolve()
                if not src_parquet_path.is_file():
                    alt = find_parquet_by_episode(src_root, src_ep_idx)
                    if alt:
                        src_parquet_path = alt
                    else:
                        raise FileNotFoundError
            except Exception:
                alt = find_parquet_by_episode(src_root, src_ep_idx)
                if alt:
                    src_parquet_path = alt
                else:
                    continue
            
            # New episode index
            new_ep_idx = int(meta_keep.info["total_episodes"])
            tgt_chunk_idx = meta_keep.get_episode_chunk(new_ep_idx)
            tgt_parquet_rel = meta_keep.data_path.format(
                episode_chunk=tgt_chunk_idx,
                episode_index=new_ep_idx
            )
            tgt_parquet_path = meta_keep.root / tgt_parquet_rel
            
            start_global_index = int(meta_keep.info.get("total_frames", 0))
            tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy and patch parquet
            try:
                df = pd.read_parquet(str(src_parquet_path))
                n_rows = len(df)
                
                def make_episode_index_col(series, scalar_val):
                    if series.dtype == object and n_rows > 0 and isinstance(series.iloc[0], (list, tuple, np.ndarray)):
                        return series.apply(lambda _: [int(scalar_val)])
                    else:
                        return pd.Series([int(scalar_val)] * n_rows, index=series.index)
                
                if "episode_index" in df.columns:
                    df["episode_index"] = make_episode_index_col(df["episode_index"], new_ep_idx)
                else:
                    df["episode_index"] = [int(new_ep_idx)] * n_rows
                
                new_global_indices = list(range(start_global_index, start_global_index + n_rows))
                if "index" in df.columns:
                    if df["index"].dtype == object and n_rows > 0 and isinstance(df["index"].iloc[0], (list, tuple, np.ndarray)):
                        df["index"] = [[int(x)] for x in new_global_indices]
                    else:
                        df["index"] = new_global_indices
                else:
                    df["index"] = new_global_indices
                
                write_parquet_like_source(df, src_parquet_path, tgt_parquet_path)
            except Exception as e:
                shutil.copy2(str(src_parquet_path), str(tgt_parquet_path))
            
            # Copy videos
            for vid_key in video_keys:
                try:
                    src_vpath_rel = src_info["video_path"].format(
                        episode_chunk=src_chunk_idx,
                        video_key=vid_key,
                        episode_index=src_ep_idx
                    )
                    src_vpath = (src_root / src_vpath_rel).resolve()
                    if not src_vpath.is_file():
                        alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                        if alt_vid:
                            src_vpath = alt_vid
                        else:
                            continue
                except Exception:
                    alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                    if alt_vid:
                        src_vpath = alt_vid
                    else:
                        continue
                
                tgt_vchunk_idx = tgt_chunk_idx
                tgt_vpath_rel = meta_keep.video_path.format(
                    episode_chunk=tgt_vchunk_idx,
                    video_key=vid_key,
                    episode_index=new_ep_idx
                )
                tgt_vpath = meta_keep.root / tgt_vpath_rel
                copy_video_tasks.append({
                    "label": f"{vid_key} episode {src_ep_idx}",
                    "src_path": str(src_vpath),
                    "tgt_path": str(tgt_vpath),
                })
            
            # Get episode stats
            ep_stats = {}
            if isinstance(src_episodes_stats, dict):
                ep_stats = src_episodes_stats.get(str(src_ep_idx), src_episodes_stats.get(src_ep_idx, {}))
            
            meta_keep.save_episode(
                episode_index=new_ep_idx,
                episode_length=ep_length,
                episode_tasks=ep_tasks,
                episode_stats=ep_stats
            )

        warnings.extend(process_video_copy_tasks(copy_video_tasks, num_workers))
        
        # Step 3: Merge extracted and kept portions
        print(f"\n[3/3] Merging extracted and kept portions...")
        if not MERGE_AVAILABLE:
            raise RuntimeError("merge_lerobot module not available. Cannot perform merge operation.")
        
        final_repo_id = repo_id + "_time_scaling"
        merge_repos(
            src_paths=[extract_tgt_path, keep_tgt_path],
            tgt_path=tgt_path,
            repo_id=final_repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            force=force
        )
        
        print(f"\n[+] Time scaling complete!")
        print(f"    Final dataset: {tgt_path}")
        print(f"    Final repo_id: {final_repo_id}")
        
    finally:
        # Cleanup temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract (downsample) frames from a LeRobot dataset"
    )
    parser.add_argument(
        "--src_path",
        required=True,
        help="Path to source LeRobot dataset"
    )
    parser.add_argument(
        "--tgt_path",
        required=True,
        help="Path to target (extracted) dataset"
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Repository ID for the new dataset"
    )
    parser.add_argument(
        "--extraction_factor",
        type=int,
        default=2,
        help="Extract every Nth frame (default: 2, meaning keep frames 0, 2, 4, ...)"
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for video processing (default: 4)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force extraction even if target directory exists"
    )
    parser.add_argument(
        "--merge_src_paths",
        nargs="+",
        default=None,
        help="Optional: Additional source dataset paths to merge with extracted dataset"
    )
    parser.add_argument(
        "--merge_tgt_path",
        type=str,
        default=None,
        help="Optional: Path for merged dataset (default: <tgt_path>_merged)"
    )
    parser.add_argument(
        "--merge_repo_id",
        type=str,
        default=None,
        help="Optional: Repository ID for merged dataset (default: <repo_id>_merged)"
    )
    parser.add_argument(
        "--merge_force",
        action="store_true",
        help="Force merge even if conflicts exist"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=None,
        help="Enable split mode: ratio of data to extract (0.0-1.0). If set, splits dataset, extracts one portion, and merges both."
    )
    
    args = parser.parse_args()
    
    # Check if split mode is enabled
    if args.split_ratio is not None:
        print("=" * 60)
        print("LeRobot Dataset Time Scaling (Split & Extract)")
        print("=" * 60)
        
        time_scaling_with_split(
            src_path=args.src_path,
            tgt_path=args.tgt_path,
            repo_id=args.repo_id,
            split_ratio=args.split_ratio,
            extraction_factor=args.extraction_factor,
            num_workers=args.num_workers,
            force=args.force
        )
    else:
        print("=" * 60)
        print("LeRobot Dataset Frame Extraction")
        print("=" * 60)
        
        extract_dataset(
            src_path=args.src_path,
            tgt_path=args.tgt_path,
            repo_id=args.repo_id,
            extraction_factor=args.extraction_factor,
            num_workers=args.num_workers,
            force=args.force,
            merge_src_paths=args.merge_src_paths,
            merge_tgt_path=args.merge_tgt_path,
            merge_repo_id=args.merge_repo_id,
            merge_force=args.merge_force
        )
