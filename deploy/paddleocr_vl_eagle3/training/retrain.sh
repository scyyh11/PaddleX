#!/usr/bin/env bash
# Re-run the clean initialization and four continuation stages on prepared corpora.
set -euo pipefail
: "${V16_DATA_ROOT:?Set V16_DATA_ROOT to the prepared corpus root}"
: "${V16_TARGET:?Set V16_TARGET to the pinned local PaddleOCR-VL-1.6 checkpoint}"
: "${V16_OUTPUT_ROOT:?Set V16_OUTPUT_ROOT to a fresh absolute output directory}"
: "${CUDA_VISIBLE_DEVICES:?Select one available GPU}"
case "$V16_OUTPUT_ROOT" in /*) ;; *) echo 'An absolute output directory is required' >&2; exit 2 ;; esac
if test -e "$V16_OUTPUT_ROOT"; then echo 'Output directory already exists' >&2; exit 2; fi
training_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
v16_python="${V16_PYTHON:-python}"
mkdir -p "$V16_OUTPUT_ROOT/models" "$V16_OUTPUT_ROOT/runs"
export OMP_NUM_THREADS=4
v16_base="$V16_DATA_ROOT/v16_clean"
"$v16_python" "$training_dir/train_clean.py" --target "$V16_TARGET" --data "$v16_base" \
  --run "$V16_OUTPUT_ROOT/runs/clean" --output "$V16_OUTPUT_ROOT/models/clean" \
  --steps 7200 --eval-every 400 --anchors-per-task 384 --anchors-per-crop 48 --batch 8 --lr .00035 --seed 20261107
v16_train=("$v16_python" -u "$training_dir/continue_train.py" --target "$V16_TARGET" --data "$v16_base")
"${v16_train[@]}" --prior "$V16_OUTPUT_ROOT/models/clean" --run "$V16_OUTPUT_ROOT/runs/aligned" --output "$V16_OUTPUT_ROOT/models/aligned" \
  --steps 4600 --eval-every 460 --validation-anchors 384 --lr .00012 --vocab-size 32768
"${v16_train[@]}" --data "$V16_DATA_ROOT/v16_external" --prior "$V16_OUTPUT_ROOT/models/aligned" --run "$V16_OUTPUT_ROOT/runs/pdf" --output "$V16_OUTPUT_ROOT/models/pdf" \
  --steps 6400 --eval-every 640 --validation-anchors 512 --lr .00025 --hard-weight .72 --temperature 1.6 --align-weight 1.2 --external-weight .83 --seed 20261120 --vocab-size 40000
"$v16_python" "$training_dir/merge_corpora.py" --input "$V16_DATA_ROOT/v16_external" "$V16_DATA_ROOT/v16_research_extra" --output "$V16_OUTPUT_ROOT/research_mix"
"${v16_train[@]}" --data "$V16_OUTPUT_ROOT/research_mix" --prior "$V16_OUTPUT_ROOT/models/pdf" --run "$V16_OUTPUT_ROOT/runs/research" --output "$V16_OUTPUT_ROOT/models/research" \
  --steps 9600 --eval-every 800 --validation-anchors 640 --lr .00016 --hard-weight .74 --temperature 1.6 --align-weight 1.2 --external-weight .87 --seed 20261121 --vocab-size 40000
"$v16_python" "$training_dir/merge_corpora.py" --input "$V16_DATA_ROOT/v16_external" "$V16_DATA_ROOT/v16_research_extra" "$V16_DATA_ROOT/v16_dense" --output "$V16_OUTPUT_ROOT/dense_mix"
"${v16_train[@]}" --data "$V16_OUTPUT_ROOT/dense_mix" --prior "$V16_OUTPUT_ROOT/models/research" --run "$V16_OUTPUT_ROOT/runs/dense" --output "$V16_OUTPUT_ROOT/models/dense" \
  --steps 8000 --eval-every 800 --validation-anchors 800 --lr .00014 --hard-weight .74 --temperature 1.6 --align-weight 1.2 --external-weight .90 --seed 20261122 --vocab-size 40000
