#!/usr/bin/env bash
set -euo pipefail

source /data/joe/anaconda3/etc/profile.d/conda.sh
conda activate vibe

CONFIG="${CONFIG:-conf/motion_tokenizer_512_large.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-exp/motion_tokenizer_512_large}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m twinlakes.bin.train_motion_tokenizer \
  --config "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
