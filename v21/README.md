v21 中存储的是lerobot v2.1 版本数据集的处理工具

# 运行环境
lerobot 0.3.3 版本，不支持 0.1.0 和 0.4.0

# 手动修改 my_env.sh 中 HF_LEROBOT_HOME 环境变量，指向数据集所在位置
# source 所有环境变量
source my_env.sh # 配置 HF_LEROBOT_HOME 环境变量 # 进入 uv venv 虚拟环境

# 可视化数据集的一个工具：https://io-ai.tech/lerobot/ # 直接将数据集拖进去即可
# rerun 可视化
    ```shell
    python -m lerobot.scripts.visualize_dataset \
        --repo-id my_Task_A/merge_kai0_advantage_b_t_std_dagger_b \
        --root /home/standard/workspace/test/kai0/data/my_Task_A/merge_kai0_advantage_b_t_std_dagger_b \
        --episode-index 0
    ```

# 0. 用小提琴图统计数据集中state和action是否有异常值
    ```shell
    # python 0_plot_lerobot_distribution.py
    python 0_plot_darwin02_distribution.py \
        --repo_id standard/darwin02_0501_2 \
        --root /home/standard/workspace/gitlab/openpi/.cache/huggingface/lerobot/standard/darwin02_0501_2 \
        --output_dir ./output

    python 0_plot_lerobot_distribution.py \
        --repo_id my_Task_A/merge_kai0_advantage_b_t_kai0_dagger_b_t_std_dagger_b_t \
        --root /home/standard/workspace/test/kai0/data/my_Task_A/merge_kai0_advantage_b_t_kai0_dagger_b_t_std_dagger_b_t \
        --output_dir ./output \
        --max_episodes 100

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

    python 5_check_dataset.py \
        dagger/piper_fold_tshirt_task_a_aligned_recodec \
        --root /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec
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
        # 1. 默认判断信号为 concat(state_key, action_key)；可用 --signal both|state|action 切换。
        #    state_key 默认自动匹配 observation.state 或 state，可用 --state_key 显式指定。
        #    action_key 默认自动匹配 action 或 actions，可用 --action_key 显式指定。
        #    输出列名默认沿用输入列名，可用 --output_state_key / --output_action_key 显式改名。
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
        # 默认视频输出为 h264_nvenc + gop=2 + b_frames=0，适合训练时随机读取 mp4 帧。
        # 若视频尺寸太小（例如 128x128），h264_nvenc 可能不支持，脚本会自动降级为 h264/libx264 保持原分辨率。
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
        #  --video_codec h264 # 强制使用 CPU libx264，适合 128x128 等 NVENC 不支持的小分辨率视频
        #  --gop 12 --b_frames 0 # 覆盖默认 GOP/B 帧设置
        #  --video_codec source --gop 2 --b_frames 0 # 沿用源编码器，但仍输出训练友好的关键帧/B帧设置

    python 11_filter_nonidle_frames.py \
        --dataset_dir /home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_0526_0603 \
        --workers 4 \
        --trim_start_seconds 0.0

    python 11_filter_nonidle_frames.py \
        --dataset_dir /home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0624 \
        --workers 8

    python 11_filter_nonidle_frames.py \
        --dataset_dir /home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion/piper_peg_and_insertion \
        --workers 8 \
        --state_key state \
        --action_key actions \
        --output_state_key observation.state \
        --output_action_key action
    # 注意，输出的name为 observation.state / action 时，https://io-ai.tech/lerobot 才能正确显示出state/action曲线。
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
    3. state/action 固定为左右臂各 7 维（每臂 6 个关节角 + 1 个夹爪），先交换左右臂，再对指定关节取反；
    4. `--negate-joints` 必须手动输入且从 1 开始计数。Piper 为 `1 4 6`，PiperX 为 `1 5 6`，脚本不会按机器人类型提供默认值；
    5. state/action key 默认从 `meta/info.json` 自动识别，也可通过 `--state-key` / `--action-key` 显式覆盖；
    6. 视频使用 FFmpeg `hflip` 滤镜重新编码，编码逻辑与 `11_filter_nonidle_frames.py` 对齐：默认 `h264_nvenc`、GOP=2、B 帧=0；
    7. 视频先写临时文件，FFprobe 校验帧数与 episode length 一致后再原子替换；任何视频失败都会终止转换；
    8. 开始写入前会打印实际采用的逐维映射、统计量公式、视频编码/映射和合并方案，便于自查。
    因此如果数据集不符合上面的要求，需要修改源码。

    完整转换方案：
    1. 输入/输出目录、键名及其来源（自动识别或参数覆盖）。
    2. 14 个输出维度逐项对应到哪个输入维度、是否取反。
    3. 视频映射：top_head -> top_head、hand_right -> hand_left、hand_left -> hand_right，均通过 FFmpeg 水平翻转并重新编码。
    4. 各统计量的变换规则、直接复制的元数据，以及 full 模式后续合并步骤。
        统计量按数学含义同步更新：mean：左右交换，指定关节取反。
        std：只左右交换，不取反。
        min/max：指定关节使用 new_min = -old_max、new_max = -old_min；其他维度正常交换。
        q01/q99：指定关节使用 new_q01 = -old_q99、new_q99 = -old_q01。
        成对统计量缺少一项时直接报错，避免生成错误统计数据。

    ```shell
    # Piper：仅生成镜像数据集
    python 13_kai0_space_mirroring.py \
        create-mirror \
        --src-path /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec \
        --tgt-path /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec_s \
        --negate-joints 1 4 6 \
        --num-workers 8

    # PiperX
    python 13_kai0_space_mirroring.py \
        create-mirror \
        --src-path /path/to/source \
        --tgt-path /path/to/mirror \
        --negate-joints 1 5 6

    # 非常规 key name 时显式指定
    python 13_kai0_space_mirroring.py \
        create-mirror \
        --src-path /path/to/source \
        --tgt-path /path/to/mirror \
        --state-key robot_state \
        --action-key robot_action \
        --negate-joints 1 4 6

    # full 子命令同样必须指定 --negate-joints；merge 子命令不需要该参数。
    # [--state-key KEY] [--action-key KEY] [--left-dim 7] [--right-dim 7] [--num-workers 4]
    # [--video-codec h264_nvenc|source|...] [--gop 2] [--b-frames 0]
    # [--nvenc-preset p4] [--nvenc-cq 23] [--av1-crf 30] [--av1-cpu-used 8]
    # [--mp4v-qscale 3] [--ffmpeg-loglevel error]

    # 使用 CPU H.264 编码；传 --video-codec source 可沿用源视频编码
    python 13_kai0_space_mirroring.py \
        create-mirror \
        --src-path /path/to/source \
        --tgt-path /path/to/mirror \
        --negate-joints 1 4 6 \
        --video-codec h264 \
        --gop 2 \
        --b-frames 0
    ```

# 14. 时间增强。kai0/train_deploy_alignment/data_augment/time_scaling.py
    ```shell
    python 14_kai0_time_scaling.py \
        --src_path /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec \
        --tgt_path /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec_t \
        --repo_id time_scaling_dataset \
        --extraction_factor 2 \
        --num-workers 8
        # --extraction_factor 2，隔帧抽，视频加速1倍
        # 默认用 FFmpeg 重新编码视频；如需沿用源编码可加 --video-codec source
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

    python 15_export_video_format_config.py \
        --dataset-dir /home/standard/workspace/test/kai0/data/my_Task_A/merge_kai0_advantage_b_t_std_dagger_b \
        --output-json ./merge_kai0_advantage_b_t_std_dagger_b.json \
        --max-videos-per-key 0


        /home/standard/workspace/test/kai0/data/my_Task_A/merge_kai0_advantage_b_t_std_dagger_b
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

# 17. 将lerobot v20转换成v21
    ```shell
    uv run --no-sync python 17_convert_dataset_v20_to_v21.py \
        --dataset_dir /home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion/collected_data/rank_0/id_0 \
        --images_to_videos \
        --output_dir /home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion/lerobot_v21
    ```

# 19. 读取 meta/info.json，自动遍历所有 dtype == "video" 的 camera mp4，检测疑似水平撕裂/错位/条带突变，并按 episode 汇总打印
    ```shell
    python 19_check_video_glitches.py \
        --dataset_dir /home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_0526_0603_nonidle \
        --workers 4 \
        --sample_stride 2 \
        --threshold 0.35 \
        --min_bad_frames 2 \
        --camera observation.images.cam_high
        # --max_episodes 5 \
    ```

# 20. 统计 LeRobot v2.1 数据集中夹爪相邻帧 delta。
脚本只读取数据集 parquet，不修改原始数据。默认检查 0 基下标 6 和 13：
left_gripper_pos / right_gripper_pos，并同时统计 observation.state 和 action。

    ```shell
    python 20_check_gripper_delta.py \
        --repo_id piperx/piperx_grab_bigbox_0526_0609_nonidle \
        --root /home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_0526_0609_nonidle \
        --output_dir ./output \
        --threshold 0.03 \
        --max_episodes 3
    ```

# 21. 读取数据集，删除 length < min_length 的 episode，并生成 <dataset>_min<min_length> 新目录，不修改原数据集。dry-run 则是只测试，不生成新数据集。
    ```shell
    python 21_remove_short_episodes.py --dry_run
    python 21_remove_short_episodes.py \
        --dataset_dir /home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0624_nonidle
    ```

# 22. 为混合数据集（全流程+dagger）的 meta/episodes.jsonl 增加 "terminated": false/true，全流程数据为true，dagger数据为false
    ```shell
    python 22_label_episode_terminated.py --dry_run
    python 22_label_episode_terminated.py \
        --dataset_dir /home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0624_nonidle_min30 \
        --terminal_start_episode 906
    ```

# 24. 查找左臂在指定位置附近连续静止超过阈值的 episode
    脚本只读取 LeRobot v2.1 数据集，不修改原始数据。默认使用 `observation.state`
    前 7 维（左臂 6 个关节和左夹爪），右臂是否运动不影响判定。默认规则为：
    关节位置误差不超过 `0.15 rad`、夹爪位置误差不超过 `0.01`、相邻帧变化不超过
    `0.001`，且连续静止时间严格超过 `2s`。诊断 JSON 默认写入 `v21/output/`。

    ```shell
    python 24_find_left_pose_idle_episodes.py

    python 24_find_left_pose_idle_episodes.py \
        --dataset_dir /home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0703_nonidle \
        --joint_position_tolerance 0.3 \
        --gripper_position_tolerance 0.01 \
        --stationary_tolerance 0.001 \
        --min_duration_seconds 2.0 \
        --output_json ./output/left_pose_idle_episodes.json

    # 剔除掉筛选出的episode
    python 1_remove_episodes.py \
        --repo_id dagger/piperx_grab_bigbox_yellow_0529_0703_nonidle \
        --root /home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0703_nonidle \
        --episodes 50,61,65,69,77,81,82,95,115,118,121,123,126,129,130,134,136,465,482,489,553,571,600,601,605,607,608,617,624,627,630,633,640,659,662,667,669,672,673,675,676,680,682,683,696,697,702,705,707,714,720,727,734,736,739,742,751,752,756,757,758,759,761,763,764,766,767,778,786,791,792,794,799,802,806,815,820,821,839,842,846,847,849,850,852,856,861,864,865,868,869,871,872,874,875,888,967,1077,1094,1315,1371,1380,1399,1417,1425,1465,1478,1485,1538,1540,1592 \
        --output_dir /home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0703_nonidle_delete
    ```

# 25. 将双 Piper 关节 state 通过 FK 转换为末端位姿，并分析空间停留分布
    脚本只读取 LeRobot v2.1 数据集，不修改原始数据。默认读取每一帧的
    `observation.state`，按 `left_joint_1_pos` ~ `left_joint_6_pos` 和
    `right_joint_1_pos` ~ `right_joint_6_pos` 分别执行 FK，夹爪维度不参与计算。
    FK 使用 AgileX 官方 `piper_sdk.C_PiperForwardKinematics` 的 DH 参数，默认启用
    关节 2、3 的 2 度补偿。位置单位为米，姿态为 RPY 弧度。

    默认末端位置是 `joint6/link6` 原点。三维图按体素累计所有 state 的帧数并换算
    停留时间；叠加散点图为了控制绘图开销会限制显示点数，但 FK、体素密度和统计
    始终使用全部 state。

    当前双臂安装默认以左臂基座为分析坐标系原点，右臂基座位于左臂 Y 轴负方向
    `0.71 m`，并假定两臂基座坐标轴方向平行。因此默认右臂基座外参为
    `0 -0.71 0 0 0 0`。

    ```shell
    # 使用脚本内置的数据集和输出目录，处理全部 episode
    python 25_plot_piper_fk_distribution.py

    # 显式指定数据集、输出目录和 1 cm 停留统计体素
    python 25_plot_piper_fk_distribution.py \
        --dataset-dir /home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0703_nonidle \
        --output-dir ./output/piper_fk_distribution \
        --voxel-size 0.01

    # 分析夹爪前端：在 joint6 局部 Z 方向增加 0.13503 m TCP 偏移
    python 25_plot_piper_fk_distribution.py \
        --dataset-dir /path/to/dataset \
        --tcp-offset 0 0 0.13503

    # 使用其他双臂安装时，可覆盖左右臂基座外参。
    # 参数顺序为 X Y Z ROLL_DEG PITCH_DEG YAW_DEG，位置单位为米。
    python 25_plot_piper_fk_distribution.py \
        --dataset-dir /path/to/dataset \
        --left-base-pose 0 0.30 0 0 0 0 \
        --right-base-pose 0 -0.30 0 0 0 0

    # 分别在两条机械臂自身的基座坐标系中分析，不应用 71 cm 安装偏移
    python 25_plot_piper_fk_distribution.py \
        --dataset-dir /path/to/dataset \
        --left-base-pose 0 0 0 0 0 0 \
        --right-base-pose 0 0 0 0 0 0

    # 快速检查前 10 个 episode，不保存体积较大的全量位姿 NPZ
    python 25_plot_piper_fk_distribution.py \
        --dataset-dir /path/to/dataset \
        --max-episodes 10 \
        --no-save-poses

    # 不生成交互 HTML，只保留静态 PNG/PDF 和统计文件
    python 25_plot_piper_fk_distribution.py \
        --dataset-dir /path/to/dataset \
        --no-interactive-html
    ```

    默认输出到 `output/piper_fk_distribution/`：

    - `end_effector_poses.npz`：每个 state 对应的左右臂完整末端位姿；
    `left_pose` / `right_pose` 的列为 `x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad`，
    并保留 `episode_index`、`frame_index` 和 `timestamp_s`。
    - `end_effector_3d_interactive.html`：可拖拽旋转、滚轮缩放、悬停查看位置和
    停留时间的交互式三维图；Plotly JS 已内嵌，离线打开即可使用。
    - `end_effector_3d_distribution.png/.pdf`：左右臂三维体素停留密度和空间叠加图。
    - `end_effector_density_projections.png/.pdf`：左右臂 XY、XZ、YZ 平面密度投影。
    - `top_occupied_voxels.csv`：停留时间最长的体素中心、帧数、秒数和占比。
    - `summary.json`：位置范围、均值、标准差、分位数、最密集体素和运行参数。

    默认叠加图和 NPZ 位姿均位于左臂基座坐标系中。若实际安装还存在旋转、高度差
    或 X 方向偏移，需要通过 `--right-base-pose` 提供完整外参。若使用旧版、未进行
    2 度补偿的 Piper DH 模型，可添加 `--no-dh-offset`。

# 26. 按闭区间抽取 episode，生成独立的 LeRobot v2.1 数据集

    `--start-episode` 和 `--end-episode` 两端都包含。例如 `10..20` 会抽取 11 个
    episode。输出中的 episode、全局帧 `index` 和使用到的 `task_index` 都会从 0
    连续重编号；`frame_index`、`timestamp` 和业务数据保持不变。

    视频按 `info.json` 的 `video_path` 模板复制并重命名，不执行 FFmpeg 重编码。
    `episodes.jsonl`、`episodes_stats.jsonl`、`tasks.jsonl` 和 `info.json` 会同步重建。
    若源数据集存在 `meta/stats.json`，会按选中 episode 重新聚合；非标准且可能失真的
    `norm_stats.json` 不会复制。

    ```shell
    # 抽取源 episode 10..20，默认输出到同级 <dataset>_ep10_20
    python 26_extract_episode_range.py \
        --dataset-dir /path/to/dataset \
        --start-episode 10 \
        --end-episode 20

    # 指定输出目录
    python 26_extract_episode_range.py \
        --dataset-dir /path/to/dataset \
        --start-episode 10 \
        --end-episode 20 \
        --output-dir /path/to/dataset_ep10_20

    # 完成全部预检并打印方案，不写入文件
    python 26_extract_episode_range.py \
        --dataset-dir /path/to/dataset \
        --start-episode 10 \
        --end-episode 20 \
        --dry-run

    # 新数据集生成并校验成功后，替换已有输出目录
    python 26_extract_episode_range.py \
        --dataset-dir /path/to/dataset \
        --start-episode 10 \
        --end-episode 20 \
        --output-dir /path/to/dataset_ep10_20 \
        --force
    ```
