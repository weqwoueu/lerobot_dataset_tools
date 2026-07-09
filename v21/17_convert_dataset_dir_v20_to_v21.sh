SRC_ROOT=/home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion_rot6d/collected_data/rank_0
DST_ROOT=/home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion_rot6d/lerobot

mkdir -p "$DST_ROOT"

for d in "$SRC_ROOT"/id_*; do
  [ -d "$d/meta" ] || continue
  name="$(basename "$d")"

  uv run --no-sync python 17_convert_dataset_v20_to_v21.py \
    --dataset_dir "$d" \
    --images_to_videos \
    --output_dir "$DST_ROOT/$name"
done