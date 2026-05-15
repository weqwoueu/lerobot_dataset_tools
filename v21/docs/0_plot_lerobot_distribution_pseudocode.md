# LeRobot 数据集分布提琴图绘制工具 — 伪代码逻辑说明

## 整体流程

```
程序入口 main():
    解析命令行参数:
        repo_id     — 数据集 repo id（必需）
        root        — 数据集本地根目录（可选）
        output_dir  — 输出图片保存目录（可选，默认当前目录）
        max_episodes — 最大处理 episode 数（可选，默认全部）
        prefix      — 输出文件名前缀（默认 "distribution"）

    确定输出目录（不存在则创建）

    dataset ← 加载数据集(repo_id, root)
    dataset_path ← dataset.root

    (特征数据字典, episode长度列表, 特征名称字典, 异常episode集合, 总episode数)
        ← 提取数据(dataset, dataset_path, max_episodes)

    打印异常数据检测报告（阈值 ±π）

    对于 特征数据字典 中的每一个 (feature_key, data_array):
        跳过无效数据

        将维度按 "左臂 / 右臂" 分组
        分别绘制左臂、右臂的提琴图并保存

    绘制 episode 长度分布提琴图并保存

    打印完成信息
```

---

## 核心函数伪代码

### 1. `load_feature_names(dataset_path, feature_key)`

从 `info.json` 读取指定特征的维度名称。

```
function load_feature_names(dataset_path, feature_key):
    info_file ← dataset_path / "meta" / "info.json"
    如果 info_file 不存在: 返回 None

    info ← 读取并解析 info_file 的 JSON
    如果 info["features"][feature_key] 存在 且 包含 "names" 字段:
        names ← 取出名称列表（处理嵌套列表情况）
        返回 names

    返回 None
```

### 2. `load_lerobot_dataset(repo_id, root)`

加载 LeRobot 格式数据集。

```
function load_lerobot_dataset(repo_id, root):
    尝试:
        dataset ← LeRobotDataset(repo_id, root=root)
        打印数据集路径、总 episode 数等信息
    失败则抛出异常

    返回 dataset
```

### 3. `extract_feature_fast(dataset, start_idx, end_idx, feature_key)`

高效提取指定范围内的特征数据。

```
function extract_feature_fast(dataset, start_idx, end_idx, feature_key):
    data_list ← []

    # 优先路径：通过 hf_dataset 批量列访问（最快）
    如果 dataset.hf_dataset 存在 且 feature_key 在其列中:
        feature_data ← hf_dataset 选取 [start_idx, end_idx) 范围的 feature_key 列
        对每个 item:
            转换为 numpy 数组 → 展平 → 追加到 data_list
        返回 data_list

    # 回退路径：逐样本索引访问
    对 idx 从 start_idx 到 end_idx:
        sample ← dataset[idx]
        如果 feature_key 在 sample 中:
            取出数据 → 转 numpy → 展平 → 追加
        否则尝试模糊匹配 key（将 "." 替换为 "_"）

    返回 data_list
```

### 4. `extract_data(dataset, dataset_path, max_episodes)`

核心数据提取函数，按 episode 遍历，提取 `action` 和 `observation.state`，并检测异常值。

```
function extract_data(dataset, dataset_path, max_episodes):
    待提取特征列表 ← ['action', 'observation.state']
    feature_data   ← { 每个特征 → 空列表 }
    feature_names  ← { 每个特征 → load_feature_names() 的结果 }

    episode_lengths     ← []
    abnormal_episodes   ← 空集合
    total_episodes      ← 0

    # ---------- 获取 episode 索引信息 ----------
    total_samples ← len(dataset)
    num_episodes  ← 从 dataset.episode_data_index 中推断 episode 数量
    如果指定了 max_episodes: num_episodes ← min(num_episodes, max_episodes)

    # ---------- 按 episode 提取（主路径） ----------
    如果 dataset.episode_data_index 可用:

        # 统一解析不同格式的 episode 索引 → episodes_indices 列表
        #   格式A: {"from": [...], "to": [...]}   → 逐对配对
        #   格式B: {ep_key: {start, end}, ...}    → 逐键解析

        对每个 (ep_id, start_idx, end_idx) in episodes_indices:
            如果 start_idx >= end_idx: 跳过

            episode_lengths 追加 (end_idx - start_idx)

            is_abnormal ← False
            对每个 feature_key in 待提取特征列表:
                data ← extract_feature_fast(dataset, start_idx, end_idx, feature_key)
                如果有数据:
                    np_data ← 转为 numpy 数组
                    追加到 feature_data[feature_key]
                    如果 np_data 中存在绝对值 > π 的元素:
                        is_abnormal ← True

            如果 is_abnormal:
                abnormal_episodes 添加 ep_id

    否则:
        # 回退：样本级提取（简化，不支持精确异常追踪）
        pass

    # ---------- 合并各 episode 数据 ----------
    final_data ← {}
    对每个 feature_key:
        检查所有片段维度一致性
        如果一致: final_data[key] ← concatenate 所有片段
        否则: 标记为 None

    返回 (final_data, episode_lengths, feature_names, abnormal_episodes, total_episodes)
```

### 5. `plot_violin_distribution(data, feature_names, title, output_path)`

绘制多维特征的提琴图（子图网格）。

```
function plot_violin_distribution(data, feature_names, title, output_path):
    如果数据为空: 直接返回

    num_features ← data 的列数
    如果 num_features > 20: 截取前 20 维

    设置绘图风格（seaborn darkgrid + husl 调色板）

    计算子图网格布局:
        n_cols ← min(4, num_features)
        n_rows ← ceil(num_features / n_cols)

    创建 (n_rows × n_cols) 子图

    对 i 从 0 到 num_features:
        取第 i 维数据
        在对应子图中绘制提琴图（含 box 内部图）
        设置标题 = 该维度名称
        设置 y 轴范围 = [min - 10%余量, max + 10%余量]

    隐藏多余子图
    调整布局

    如果指定了 output_path: 保存为 PNG (300 dpi)
    否则: plt.show()
```

### 6. `plot_episode_length_distribution(episode_lengths, output_path)`

绘制 episode 长度的分布提琴图。

```
function plot_episode_length_distribution(episode_lengths, output_path):
    如果列表为空: 返回

    创建单个提琴图，y 轴 = episode 长度（帧数）
    设置标题 "Episode Length Distribution"

    保存或显示图表
```

---

## 主流程中的左右臂分组逻辑

在 `main()` 中绘制每个特征的提琴图前，会对维度进行左/右分组：

```
如果 info.json 中有维度名称 且 数量匹配:
    对每个维度名称:
        简化名称（取末两级路径）
        如果名称包含 "Left" / "masterLeft" / "pikaSensor_l" / "pikaGripper_l":
            归入左臂组
        如果名称包含 "Right" / "masterRight" / "pikaSensor_r" / "pikaGripper_r":
            归入右臂组
        否则:
            按维度序号前半 → 左臂，后半 → 右臂
否则:
    按维度数平分：前半 → 左臂，后半 → 右臂

分别对左臂数据、右臂数据各生成一张提琴图
```

---

## 数据流概览

```
命令行参数
    │
    ▼
LeRobotDataset 加载
    │
    ▼
info.json → 读取维度名称
    │
    ▼
按 episode 遍历 ──► extract_feature_fast()
    │                   ├─ 优先 hf_dataset 批量读取
    │                   └─ 回退 dataset[idx] 逐条读取
    │
    ├─ 收集 action 数据矩阵
    ├─ 收集 observation.state 数据矩阵
    ├─ 记录每个 episode 长度
    └─ 检测异常 episode（|值| > π）
         │
         ▼
    合并 & 维度校验
         │
         ├──► 左/右臂分组
         │       ├─ 左臂提琴图 → PNG
         │       └─ 右臂提琴图 → PNG
         │
         ├──► episode 长度提琴图 → PNG
         │
         └──► 异常检测报告（控制台输出）
```
