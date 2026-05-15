#!/usr/bin/env python3
"""诊断 LeRobot 本地数据集完整性的脚本。

用法:
    python scripts/check_dataset.py <repo_id> [--root <root_dir>]

示例:
    python scripts/check_dataset.py my_Task_A/advantage_dagger
    python scripts/check_dataset.py my_Task_A/advantage_dagger --root /path/to/custom/cache
"""

import argparse
import json
import os
import sys
from pathlib import Path


def get_default_lerobot_home() -> Path:
    return Path(os.environ.get("LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"))


def check_dataset(repo_id: str, root: Path | None = None):
    lerobot_home = root or get_default_lerobot_home()
    dataset_dir = lerobot_home / repo_id

    print("=" * 70)
    print(f"  LeRobot 数据集诊断工具")
    print("=" * 70)
    print(f"\n  LEROBOT_HOME : {lerobot_home}")
    print(f"  repo_id      : {repo_id}")
    print(f"  数据集路径    : {dataset_dir}")
    print()

    # ── 1. 检查目录是否存在 ──
    print("─" * 70)
    print("[1/6] 检查数据集目录是否存在...")
    if not dataset_dir.exists():
        print(f"  ❌ 目录不存在: {dataset_dir}")
        print(f"\n  可能的原因:")
        print(f"    - 数据集没有被正确上传/拷贝到这台机器")
        print(f"    - LEROBOT_HOME 环境变量设置不正确 (当前: {lerobot_home})")
        print(f"    - repo_id 不正确 (当前: {repo_id})")
        print(f"\n  请检查以下目录是否有你的数据集:")
        print(f"    ls {lerobot_home}/")
        return False

    print(f"  ✅ 目录存在")
    print(f"\n  目录内容:")
    for item in sorted(dataset_dir.iterdir()):
        if item.is_dir():
            sub_count = len(list(item.iterdir()))
            print(f"    📁 {item.name}/ ({sub_count} 个文件)")
        else:
            size = item.stat().st_size
            print(f"    📄 {item.name} ({size:,} bytes)")

    # ── 2. 检查 meta/info.json ──
    print()
    print("─" * 70)
    print("[2/6] 检查 meta/info.json ...")
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        print(f"  ❌ 文件不存在: {info_path}")
        print(f"  这是 LeRobot v2 数据集的核心元数据文件，缺失说明数据集格式不正确。")
        return False

    with open(info_path) as f:
        info = json.load(f)

    total_episodes = info.get("total_episodes", "未知")
    total_frames = info.get("total_frames", "未知")
    total_chunks = info.get("total_chunks", "未知")
    chunks_size = info.get("chunks_size", "未知")
    fps = info.get("fps", "未知")
    data_path = info.get("data_path", "未知")
    video_path = info.get("video_path", "未知")
    print(f"  ✅ 文件存在，内容:")
    print(f"    total_episodes : {total_episodes}")
    print(f"    total_frames   : {total_frames}")
    print(f"    total_chunks   : {total_chunks}")
    print(f"    chunks_size    : {chunks_size}")
    print(f"    fps            : {fps}")
    print(f"    data_path      : {data_path}")
    if video_path != "未知":
        print(f"    video_path     : {video_path}")

    features = info.get("features", {})
    if features:
        print(f"    features       : {list(features.keys())}")

    # ── 3. 检查 meta/episodes.jsonl ──
    print()
    print("─" * 70)
    print("[3/6] 检查 meta/episodes.jsonl ...")
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        print(f"  ❌ 文件不存在: {episodes_path}")
        return False

    episodes = []
    with open(episodes_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
                episodes.append(ep)
            except json.JSONDecodeError as e:
                print(f"  ❌ 第 {line_num} 行 JSON 解析失败: {e}")
                print(f"     内容: {line[:200]}")
                return False

    print(f"  ✅ 文件存在，共 {len(episodes)} 条 episode 记录")
    if episodes:
        print(f"    第一条: {episodes[0]}")
        if len(episodes) > 1:
            print(f"    最后一条: {episodes[-1]}")

    if isinstance(total_episodes, int) and len(episodes) != total_episodes:
        print(f"  ⚠️  警告: info.json 中 total_episodes={total_episodes}，"
              f"但 episodes.jsonl 中有 {len(episodes)} 条记录，不一致!")

    # ── 4. 检查 data/ 目录下的 parquet 文件 ──
    print()
    print("─" * 70)
    print("[4/6] 检查数据文件 (parquet) ...")

    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        print(f"  ❌ data/ 目录不存在: {data_dir}")
        return False

    actual_parquets = sorted(data_dir.rglob("*.parquet"))
    print(f"  找到 {len(actual_parquets)} 个 parquet 文件。")
    
    empty_parquets = []
    for p in actual_parquets:
        size = p.stat().st_size
        if size == 0:
            empty_parquets.append(p.relative_to(dataset_dir))
            
    if empty_parquets:
        print(f"  ⚠️  发现 {len(empty_parquets)} 个大小为 0 的空 parquet 文件:")
        for ep in empty_parquets[:10]:
            print(f"     ⚠️  {ep}")
        if len(empty_parquets) > 10:
            print(f"     ... 还有 {len(empty_parquets) - 10} 个未列出")

    # 根据 info.json 推算期望的文件
    if isinstance(total_episodes, int) and isinstance(chunks_size, int):
        expected_files = []
        dp = data_path if data_path != "未知" else "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        for ep_idx in range(total_episodes):
            chunk_idx = ep_idx // chunks_size
            fpath = dp.format(episode_chunk=chunk_idx, episode_index=ep_idx)
            expected_files.append(fpath)

        print(f"  根据 info.json 推算，期望有 {len(expected_files)} 个数据文件:")

        missing = []
        for fpath in expected_files:
            full_path = dataset_dir / fpath
            if not full_path.exists():
                missing.append(fpath)

        if missing:
            print(f"  ❌ 缺失 {len(missing)} 个文件:")
            for m in missing[:20]:
                print(f"     ❌ {m}")
            if len(missing) > 20:
                print(f"     ... 还有 {len(missing) - 20} 个未列出")
        else:
            print(f"  ✅ 所有期望的数据文件都存在")

    # ── 4.5 检查 videos/ 目录下的 mp4 文件 ──
    print()
    print("─" * 70)
    print("[4.5/6] 检查视频文件 (mp4) ...")
    
    videos_dir = dataset_dir / "videos"
    if not videos_dir.exists():
        print(f"  ⚠️  videos/ 目录不存在: {videos_dir} (如果你的数据集不包含视频，可以忽略)")
    else:
        actual_videos = sorted(videos_dir.rglob("*.mp4"))
        print(f"  找到 {len(actual_videos)} 个 mp4 文件:")
        
        # 检查是否有大小为 0 的视频文件
        empty_videos = []
        for v in actual_videos:
            size = v.stat().st_size
            if size == 0:
                empty_videos.append(v.relative_to(dataset_dir))
        
        if empty_videos:
            print(f"  ⚠️  发现 {len(empty_videos)} 个大小为 0 的空视频文件:")
            for ev in empty_videos[:10]:
                print(f"     ⚠️  {ev}")
            if len(empty_videos) > 10:
                print(f"     ... 还有 {len(empty_videos) - 10} 个未列出")
        
        # 根据 info.json 推算期望的视频文件
        if isinstance(total_episodes, int) and isinstance(chunks_size, int) and isinstance(info.get("features"), dict):
            # 找出所有的视频特征键 (例如 observation.images.top_head)
            video_keys = []
            for key, feature in info["features"].items():
                if isinstance(feature, dict) and feature.get("dtype") == "video":
                    video_keys.append(key)
            
            if video_keys:
                print(f"\n  根据 info.json 推算，包含视频的特征有: {video_keys}")
                expected_videos = []
                for ep_idx in range(total_episodes):
                    chunk_idx = ep_idx // chunks_size
                    for key in video_keys:
                        # 如果 info.json 提供了 video_path，优先使用它
                        if video_path != "未知":
                            # video_path 例如 "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
                            v_name = video_path.format(
                                video_key=key, 
                                episode_chunk=chunk_idx, 
                                episode_index=ep_idx
                            )
                            expected_videos.append(v_name)
                        else:
                            print("  ❌ info.json 中未提供 video_path，无法推算视频路径。")
                            return False
                
                print(f"  期望有 {len(expected_videos)} 个视频文件:")
                
                missing_videos = []
                for vpath in expected_videos:
                    full_path = dataset_dir / vpath
                    if not full_path.exists():
                        missing_videos.append(vpath)
                
                if missing_videos:
                    print(f"  ❌ 缺失 {len(missing_videos)} 个视频文件:")
                    for m in missing_videos[:20]:
                        print(f"     ❌ {m}")
                    if len(missing_videos) > 20:
                        print(f"     ... 还有 {len(missing_videos) - 20} 个未列出")
                else:
                    print(f"  ✅ 所有期望的视频文件都存在")
            else:
                print(f"  ℹ️  info.json 的 features 中没有声明 video 类型的特征")

    # ── 5. 检查 meta/tasks.jsonl ──
    print()
    print("─" * 70)
    print("[5/6] 检查 meta/tasks.jsonl ...")
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    tasks_in_meta = {}
    if not tasks_path.exists():
        print(f"  ⚠️  文件不存在: {tasks_path}")
        print(f"  (部分数据集可能不需要此文件)")
    else:
        tasks = []
        with open(tasks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    tasks.append(item)
                    if "task_index" in item and "task" in item:
                        tasks_in_meta[item["task_index"]] = item["task"]
        print(f"  ✅ 文件存在，共 {len(tasks)} 个 task")
        for t in tasks[:10]:
            print(f"    {t}")

    # ── 5.5 检查 parquet 中的 task_index 是否与 tasks.jsonl 一致 ──
    print()
    print("─" * 70)
    print("[5.5/6] 检查 parquet 中的 task_index 是否与 tasks.jsonl 一致 ...")
    try:
        import pandas as pd
        
        print("  正在读取所有 parquet 文件提取 task_index (这可能需要一些时间)...")
        all_task_indices = set()
        file_task_map = {}
        
        # actual_parquets 在第 4 步已经获取到了
        if 'actual_parquets' in locals() and actual_parquets:
            for pf in actual_parquets:
                try:
                    df = pd.read_parquet(pf, columns=["task_index"])
                    unique_indices = set(df["task_index"].unique().tolist())
                    all_task_indices.update(unique_indices)
                    file_task_map[str(pf.relative_to(dataset_dir))] = unique_indices
                except Exception as e:
                    # 有些 parquet 可能没有 task_index 列
                    pass
            
            print(f"  parquet 文件中包含的 task_index: {sorted(all_task_indices)}")
            
            if tasks_in_meta:
                missing = all_task_indices - set(tasks_in_meta.keys())
                extra = set(tasks_in_meta.keys()) - all_task_indices
                
                if missing:
                    print(f"  ❌ 错误: parquet 中存在 tasks.jsonl 未定义的 task_index: {sorted(missing)}")
                    for fname, indices in file_task_map.items():
                        file_missing = indices & missing
                        if file_missing:
                            print(f"     {fname} 包含: {sorted(file_missing)}")
                    print(f"  💡 提示: 可以使用 tools/lerobot_dataset_tools/3_fix_task_index.py 修复此问题")
                else:
                    print(f"  ✅ 所有 parquet 中的 task_index 都已在 tasks.jsonl 中定义")
                    
                if extra:
                    print(f"  ℹ️  提示: tasks.jsonl 中定义了但未使用的 task_index: {sorted(extra)}")
            else:
                if all_task_indices:
                    print(f"  ⚠️  parquet 中包含 task_index {sorted(all_task_indices)}，但 tasks.jsonl 不存在或为空")
        else:
            print("  ⚠️  未找到 parquet 文件，跳过检查")
            
    except ImportError:
        print("  ⚠️  无法导入 pandas，跳过 task_index 检查")
        print("  请安装 pandas: pip install pandas")

    # ── 6. 尝试用 LeRobot 加载 ──
    print()
    print("─" * 70)
    print("[6/6] 尝试用 LeRobot 加载数据集 ...")
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    except ImportError:
        print("  ⚠️  无法导入 lerobot，跳过加载测试")
        print("  请确保 lerobot 已安装: pip install lerobot")
        return True

    load_kwargs = {"local_files_only": True}
    if root is not None:
        load_kwargs["root"] = root
    print(f"  加载参数: {load_kwargs}")

    try:
        print(f"\n  加载 LeRobotDatasetMetadata('{repo_id}') ...")
        # 尝试直接构造本地路径
        local_dir = root / repo_id if root else get_default_lerobot_home() / repo_id
        if local_dir.exists():
            print(f"  尝试直接从本地路径加载: {local_dir}")
            # LeRobotDatasetMetadata 接受本地绝对路径作为 repo_id
            meta = LeRobotDatasetMetadata(str(local_dir))
        else:
            meta = LeRobotDatasetMetadata(repo_id, **load_kwargs)
            
        print(f"  ✅ Metadata 加载成功")
        print(f"    fps    : {meta.fps}")
        print(f"    tasks  : {meta.tasks}")
        print(f"    features: {list(meta.features.keys()) if hasattr(meta, 'features') else 'N/A'}")
    except TypeError as e:
        print(f"  ⚠️  LeRobotDatasetMetadata 不支持 local_files_only 参数，尝试不带该参数...")
        print(f"    ({type(e).__name__}: {e})")
        load_kwargs.pop("local_files_only", None)
        try:
            meta = LeRobotDatasetMetadata(repo_id, **load_kwargs)
            print(f"  ✅ Metadata 加载成功 (无 local_files_only)")
            print(f"    fps    : {meta.fps}")
            print(f"    tasks  : {meta.tasks}")
        except Exception as e2:
            print(f"  ❌ Metadata 加载仍然失败:")
            print(f"    {type(e2).__name__}: {e2}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"  ❌ Metadata 加载失败:")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        print(f"\n  ── 尝试直接从本地文件构造 Metadata ──")
        print(f"  检查 LeRobotDatasetMetadata 的构造函数签名...")
        import inspect
        sig = inspect.signature(LeRobotDatasetMetadata.__init__)
        print(f"  签名: LeRobotDatasetMetadata{sig}")
        return False

    try:
        print(f"\n  加载 LeRobotDataset('{repo_id}') ...")
        local_dir = root / repo_id if root else get_default_lerobot_home() / repo_id
        if local_dir.exists():
            print(f"  尝试直接从本地路径加载: {local_dir}")
            # LeRobotDataset 同样接受本地绝对路径
            dataset = LeRobotDataset(str(local_dir))
        else:
            dataset = LeRobotDataset(repo_id, **load_kwargs)
            
        print(f"  ✅ Dataset 加载成功!")
        print(f"    长度: {len(dataset)}")
        print(f"    episodes: {dataset.episodes}")

        print(f"\n  尝试读取第一条数据...")
        sample = dataset[0]
        print(f"  ✅ 第一条数据读取成功!")
        print(f"    keys: {list(sample.keys())}")
        for k, v in sample.items():
            if hasattr(v, "shape"):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, str):
                print(f"    {k}: '{v[:100]}'")
            else:
                print(f"    {k}: {type(v).__name__} = {v}")

    except TypeError as e:
        print(f"  ⚠️  LeRobotDataset 不支持某些参数，打印签名以供排查...")
        print(f"    ({type(e).__name__}: {e})")
        import inspect
        sig = inspect.signature(LeRobotDataset.__init__)
        print(f"  签名: LeRobotDataset{sig}")
        return False
    except AssertionError as e:
        print(f"  ❌ Dataset 加载失败 (AssertionError):")
        print(f"    {e}")
        print(f"\n  这通常意味着 episodes.jsonl 中声明的某些 episode 对应的 parquet 文件缺失。")
        print(f"  请对比上面 [4/6] 中列出的实际文件和期望文件。")
        return False
    except Exception as e:
        print(f"  ❌ Dataset 加载失败:")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 70)
    print("  ✅ 所有检查通过，数据集看起来是完整的!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查 LeRobot 本地数据集的完整性")
    parser.add_argument("repo_id", help="数据集的 repo_id，例如 my_Task_A/advantage_dagger")
    parser.add_argument("--root", type=Path, default=None,
                        help="自定义 LEROBOT_HOME 路径 (默认: ~/.cache/huggingface/lerobot)")
    args = parser.parse_args()

    ok = check_dataset(args.repo_id, args.root)
    sys.exit(0 if ok else 1)
