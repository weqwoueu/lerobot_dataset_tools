# 设置uv走国内源
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
# 设置huggingface走国内镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 服务器上根目录（系统盘）下空间很少，而python环境、数据集、ckpt文件都非常大，得手动把默认路径（在系统盘下）重定向到数据盘（/home/public-4T）下。

# 设置uv缓存python包的路径为本地路径，而非默认路径（/home/${user}/.cache/uv）
export UV_CACHE_DIR=$(pwd)/.cache/uv
# 设置huggingface lerobot数据集存储在本地路径，而非系统默认路径（/home/${user}/.cache/huggingface/lerobot）
export HF_LEROBOT_HOME=/home/standard/workspace/gitlab/lerobot_dataset_tools/v21/.cache/huggingface/lerobot

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
