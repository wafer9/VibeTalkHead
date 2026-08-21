#!/bin/bash

. ./path.sh || exit 1;
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="4"

dir=exp/s5_1p7_all_semantic
test=hdtf
checkpoints='8'
cfg_scales='3.0'

for checkpoint in ${checkpoints}; do
{
  for scale in ${cfg_scales}; do
    {
      python -m twinlakes.bin.infer_video \
        --config ${dir}/train.yaml \
        --checkpoint ${dir}/${checkpoint}.pt \
        --test_data data/${test}/test.jsonl \
        --result_dir ${dir}/infer_${test}_${checkpoint}_${scale} \
        --num_steps 10 --cfg_scale ${scale}  --limit 100
    }
  done
}
done