v21 中存储的是lerobot v2.1 版本数据集的处理工具

# 可视化数据集的一个工具：https://io-ai.tech/lerobot/ # 直接将数据集拖进去即可

# 运行环境
lerobot 0.3.3 版本，不支持 0.1.0 和 0.4.0

# 手动修改 my_env.sh 中 HF_LEROBOT_HOME 环境变量，指向数据集所在位置
# source 所有环境变量
source my_env.sh # 配置 HF_LEROBOT_HOME 环境变量 # 进入 uv venv 虚拟环境


# 0. 用小提琴图统计数据集中state和action是否有异常值
    ```shell
    # python 0_plot_lerobot_distribution.py
    python 0_plot_darwin02_distribution.py \
        --repo_id standard/darwin02_0501_2 \
        --root /home/standard/workspace/gitlab/openpi/.cache/huggingface/lerobot/standard/darwin02_0501_2 \
        --output_dir ./output

    # 会输出异常值所在的episode，和异常比例。
    ```

    ### QA
    ```
    提琴图与 KDE
    提琴图本质上就是 KDE。所有常见的提琴图实现（seaborn、matplotlib、ggplot2、plotly）都使用核密度估计来绘制，这不是某种特殊选项，而是提琴图的定义本身：
    > 提琴图 = 箱线图 + 两侧对称的 KDE 密度曲线
    KDE 用高斯核对每个数据点做平滑，叠加后得到连续的概率密度估计曲线，曲线越宽表示该值附近的数据越密集。
    对连续数据这很合理，比如关节角在 [-π, π] 范围内有丰富的连续取值，KDE 能准确反映分布形状。
    对离散数据就会产生误导，因为 KDE 的高斯核会把 0 和 1 两个点向中间"扩散"，生成 0~1 之间看似有数据的假象。这是 KDE 方法的固有局限，不是 bug。
    应对离散数据的常见做法
    方法	适用场景
    柱状图 / 条形图（bar chart）	二值、少量离散值，直接显示各值的计数或占比
    计数图（countplot）	类别型数据
    调小带宽 bw_adjust=0.1	让 KDE 更"尖锐"，但仍是连续曲线
    cut=0	限制 KDE 不超出数据范围，但 0~1 之间仍有平滑
    抖动散点图（stripplot）	少量数据点时直观
    当前脚本已经做了自动检测：unique 值 <= 5 个的维度用柱状占比图，其余用提琴图，两种数据类型各得其所。
    ```

# 1. 删除包含异常值的episode，得到新的数据集
    ```shell
    python 1_remove_episodes.py \
        --repo_id standard/fold_towel \
        --root /path/to/dataset/{dataset_name} \
        --episodes 0,1,2 \
        --output_dir ./tools/output/{dataset_name}
    ```

# 2. 合并n个lerobot v2.1的数据集
    ```shell
    # 参考自 https://github.com/jackvial/lerobot-data-studio
    # 支持多个v21版本的lerobot数据集的合并。

    # 手动将需要合并的数据集填入 2_merge_multi_v21_lerobot_datasets.py 的 list 中
    python 2_merge_multi_v21_lerobot_datasets.py


    # 金山云 xingliu-01
    cd /mnt/kpfs/heyang/openpi
    source my_env.sh
    deactivate # 退出openpi的uv环境
    conda deactivate # 退出base conda环境
    conda activate lerobot # lerobot  0.3.3  /mnt/kpfs/heyang/lerobot
    cd /mnt/kpfs/heyang/openpi/tools/merge_multi_v21_lerobot_datasets
    # 云上没使用uv，而是用了conda，另外只有xingliu-01装了conda lerobot环境
    ```

# 3. 一个临时的工具，用来手动调整数据集中parquet的task_index
    ```shell
    # piperx_ctrl_fold_towel_long 数据集是单任务的，但上游失误，塞了两个类似的task_prompt，需要改成一个，且同步调整parquet中的task_index值。
    python 3_fix_task_index.py \
        --dataset_dir /mnt/kpfs/heyang/openpi/.cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long
    # 2026-02-13 09:27:29,138 - __main__ - INFO - ============================================================
    # 2026-02-13 09:27:29,139 - __main__ - INFO - 数据集 task_index 检查报告
    # 2026-02-13 09:27:29,139 - __main__ - INFO - ============================================================
    # 2026-02-13 09:27:29,139 - __main__ - INFO - 
    # tasks.jsonl 中定义的任务 (1 个):
    # 2026-02-13 09:27:29,139 - __main__ - INFO -   task_index=0: "Take one towel from the basket, spread it flat, fold it twice, and then stack the folded towels together."
    # 2026-02-13 09:27:29,139 - __main__ - INFO - 
    # parquet 文件中出现的 task_index: [0, 1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING - 
    # ⚠️  缺失的 task_index (在 parquet 中存在但 tasks.jsonl 中未定义): [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000021.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000022.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000023.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000024.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000025.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000026.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000027.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000028.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000029.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000030.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000031.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000032.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000033.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - WARNING -    data/chunk-000/episode_000034.parquet 中包含缺失的 task_index: [1]
    # 2026-02-13 09:27:29,139 - __main__ - INFO - ============================================================
    # 2026-02-13 09:27:29,146 - __main__ - INFO - 已备份 tasks.jsonl -> /mnt/kpfs/heyang/openpi/.cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long/meta/backup/tasks.jsonl.bak
    # 2026-02-13 09:27:29,149 - __main__ - INFO - 已备份 episodes.jsonl -> /mnt/kpfs/heyang/openpi/.cache/huggingface/lerobot/standard/piperx_ctrl_fold_towel_long/meta/backup/episodes.jsonl.bak
    # 2026-02-13 09:27:29,309 - __main__ - INFO -   已修复 episode_000021.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,344 - __main__ - INFO -   已修复 episode_000022.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,372 - __main__ - INFO -   已修复 episode_000023.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,402 - __main__ - INFO -   已修复 episode_000024.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,436 - __main__ - INFO -   已修复 episode_000025.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,463 - __main__ - INFO -   已修复 episode_000026.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,494 - __main__ - INFO -   已修复 episode_000027.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,524 - __main__ - INFO -   已修复 episode_000028.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,552 - __main__ - INFO -   已修复 episode_000029.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,581 - __main__ - INFO -   已修复 episode_000030.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,618 - __main__ - INFO -   已修复 episode_000031.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,646 - __main__ - INFO -   已修复 episode_000032.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,676 - __main__ - INFO -   已修复 episode_000033.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,704 - __main__ - INFO -   已修复 episode_000034.parquet: task_index [1] -> [0]
    # 2026-02-13 09:27:29,864 - __main__ - INFO - 共修复了 14 个 parquet 文件
    # 2026-02-13 09:27:29,867 - __main__ - INFO -   已更新 tasks.jsonl: 仅保留 task_index=0, task="Take one towel from the basket, spread it flat, fold it twice, and then stack the folded towels together."
    # 2026-02-13 09:27:29,872 - __main__ - INFO -   已更新 episodes.jsonl 中的 task_index 为 0
    # 2026-02-13 09:27:29,872 - __main__ - INFO - 
    # ✅ 统一模式修复完成！所有 task_index 已统一为 0
    ```

# 5. 检查数据集完整性，会输出有异常的episode index，后续可以用其他工具剔除有误的episode并导出完整的新数据集
    ```shell
    python 5_check_dataset.py \
        dagger/piper_fold_tshirt_green_small \
        --root /home/standard/workspace/test/kai0/data/standard_Task_A
    # 查出245和474数据集缺少数据，到https://io-ai.tech/lerobot上加载数据集，然后删除掉缺失的episode，再导出到本地即可
```

# 6. darwin02数据集，将首帧的六维力传感器的force部分作为offset，修正episode中其后的每一帧
    ```shell
    python 6_convert_force_to_delta.py \
        --repo_id standard/darwin02_0503_trim_tail \
        --root /home/standard/workspace/gitlab/openpi/.cache/huggingface/lerobot/standard/darwin02_0503_trim_tail \
        --output_dir /home/standard/workspace/gitlab/openpi/.cache/huggingface/lerobot/standard/darwin02_0503_trim_tail_force_delta
    ```

# 8. 批量转换 videos/ 下 mp4 文件的视频编码
    ```shell
    # 将 av1 转成 mp4v
    python 8_convert_video_codec.py \
        --dataset_dir .cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human \
        --target_codec mp4v \
        --source_codec av1

    # 将 mp4v 转回 av1
    python 8_convert_video_codec.py \
        --videos_dir .cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human/videos \
        --target_codec av1 \
        --source_codec mp4v
    ```

# 9. 剔除指定数据集的最后1s，20帧
    ```shell
    # darwin02_0503数据集的最后1s数据是脏数据（混入了部分场景恢复的数据），需要剔除
    python 9_trim_tail_frames.py \
    --dataset_dir /home/standard/workspace/gitlab/openpi/.cache/huggingface/lerobot/standard/darwin02_0503 \
    --output_dir /home/standard/workspace/gitlab/openpi/.cache/huggingface/lerobot/standard/darwin02_0503_trim_tail
    ```

# 10. 源数据集的mp4编码时gop参数编码错误，用该工具重编码（非原地），生成新数据集，并把旧 mp4 备份到 output_dir/old_video/
    ```shell
    # 检测数据集视频 GOP/关键帧间隔
    # 默认只抽查前 10 个视频；如需全量检查可加 --max_check_videos 0
    # 关键帧间隔过长，会导致dataloader随机读取视频帧耗时更长，导致训练时，gpu等待batch数据
    python 10_reencode_video_gop.py \
        --dataset_dir .cache/huggingface/lerobot/standard/darwin02_0503_trim_tail_force_delta \
        --check

    python 10_reencode_video_gop.py \
        --dataset_dir .cache/hsuggingface/lerobot/standard/darwin02_0503_trim_tail_force_delta \
        --output_dir .cache/huggingface/lerobot/standard/darwin02_0503_trim_tail_force_delta_gop2 \
        --gop 2 \
        --workers 16
    ```

# 11. 剔除lerobot数据集中的静止帧，和首尾不需要的指定帧。如果使用robocoin采集的数据集，需要用这个工具进行剔除。
    ```shell
    # kai0，piper双臂叠衣服任务的数据集。
    # 剔除静止帧的逻辑：
        # 1. 默认判断信号为 concat(observation.state, action)，即 28 维；可用 --signal both|state|action 切换。
        # 2. 相邻帧比较：
            # 对第 t 帧，计算 abs(signal[t] - signal[t-1])。
            # 如果所有维度变化都 < eps，默认 eps=1e-3，则第 t 帧视为静止帧。
            # 第 0 帧默认不视为静止帧。
        # 3. 连续静止段过滤：
            # 只有连续静止长度 >= min_idle_len 才删除，默认 min_idle_len=7。
            # 如果只是短暂停顿，例如连续静止 3 帧，默认不会删。
        # 4. 非静止片段长度过滤：
            # 删除长静止段后，会得到若干个保留片段。
            # 如果某个保留片段长度 < min_nonidle_len，默认 16，这个片段也会整体丢弃。
            # 也就是说，短于 16 帧的“运动小碎片”默认不保留，因为太短，不适合作为训练采样片段。
        # 4. 片段末尾裁剪：
            # filter_last_n=10 不是指每个 episode 最后删 10 帧。
            # 它是指 每一个保留下来的非静止片段 的末尾都再裁掉 10 帧。
            # 例如某个 episode 中保留下来两个非静止片段 [20, 80) 和 [150, 230)，最终会变成 [20, 70) 和 [150, 220)。
            # 这样做是 OpenPI/DROID 风格，目的是避免采到动作 chunk 尾部很多静止动作的位置。
        # 若某个 episode 最终没有任何保留帧，脚本默认跳过该 episode，并在统计里记录。
    python 11_filter_nonidle_frames.py \
        --dataset_dir /home/standard/workspace/test/kai0/data/standard_Task_A/base/piper_fold_tshirt_red \
        --workers 4 \
        --trim_start_seconds 2.0
        # --trim_start_seconds 2.0 # 最初的dagger数据集开头肯能会因为状态切换，机械臂发生剧烈抖动，需要裁剪掉，大致2s即可。按秒数裁剪
        # --trim_end_seconds 0.5
        # --trim_start_frames 0 # 按帧数裁剪
        # --trim_end_frames 0
        #   --dry_run # 预览
        #  --video_codec source # 严格沿用原lerobot数据集相同的视频编码格式。否则默认用h264_nvenc，在4090上编码速度快
    ```

# 12. kai0 的临时工具。用robocoin采的数据集，key name等参数和kai0的数据集不一致，用该工具对齐，除了视频编码方式不对齐。
    ```shell
    # 将 piper 叠衣服数据集对齐到 Task_A/dagger 的 LeRobot v2.1 元数据格式。
    # 默认不会修改源数据集，而是生成一个新的输出目录。脚本只做 key/prompt/meta
    # 对齐和文件复制，不转码视频，也不改写 parquet 内容。
    python 12_convert_piper_to_task_a_dagger.py \
        --source_dir /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt \
        --target_reference_dir /home/standard/workspace/test/kai0/data/Task_A/dagger \
        --dry_run

    python 12_convert_piper_to_task_a_dagger.py \
        --source_dir /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt \
        --target_reference_dir /home/standard/workspace/test/kai0/data/Task_A/dagger \
        --output_dir /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned
    ```

# 13. 空间增强。kai0/train_deploy_alignment/data_augment/space_mirroring.py
    注意，代码中：
    1. 对 observation.images.top_head / observation.images.hand_left / observation.images.hand_right 下的视频进行翻转；
    2. 交换 observation.images.hand_left / observation.images.hand_right 两个文件夹；
    因此如果数据集不符合上面的要求，需要修改源码。

    ```shell
    # 仅生成镜像的数据集
    python 13_kai0_space_mirroring.py \
        create-mirror \
        --src-path /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec \
        --tgt-path /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec_s \
        --num-workers 16
    #   [--fps 30] [--robot-type agilex] [--left-dim 7] [--right-dim 7] [--num-workers 4] [--features-json /path/to/features.json] [--force]
    ```

# 14. 时间增强。kai0/train_deploy_alignment/data_augment/time_scaling.py
    ```shell
    python 14_kai0_time_scaling.py \
        --src_path /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec \
        --tgt_path /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec_t \
        --repo_id time_scaling_dataset \
        --extraction_factor 2 \
        --num-workers 16
        # --extraction_factor 2，隔帧抽，视频加速1倍
    ```

# 15. 导出数据集视频编码参数配置，供后续工具按指定参数生成视频
    ```shell
    # 默认每个 video key 均匀抽查 10 个视频，打印摘要并写 JSON 配置
    python 15_export_video_format_config.py \
        --dataset-dir /home/standard/workspace/test/kai0/data/Task_A/advantage \
        --output-json ./video_format_config.json

    # 全量探测所有视频
    python 15_export_video_format_config.py \
        --dataset-dir /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned \
        --output-json ./piper_fold_tshirt_task_a_aligned_video_format_config_full.json \
        --max-videos-per-key 0
    ```

# 16. 根据视频参数配置重编码数据集视频，非原地生成 _recodec 新数据集
    ```shell
    python 16_recodec_dataset_with_video_config.py \
        --dataset-dir /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned \
        --config-json ./video_format_config.json \
        --workers 16

    # 默认输出到输入数据集同级目录: <dataset_name>_recodec
    # 可用 --output-dir 指定输出目录；可用 --dry-run 只打印计划。
    ```

# 17. 将lerobot v20转换成v21（未实际测试）
    ```shell
    python 17_convert_dataset_v20_to_v21.py
    ```
