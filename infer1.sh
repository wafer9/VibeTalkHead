#!/bin/bash

. ./path.sh || exit 1;
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="6"

dir=exp/s2
test=hdtf
checkpoints='1_45000'
cfg_scales='3.0'

for checkpoint in ${checkpoints}; do
{
  for scale in ${cfg_scales}; do
    {
      python -m twinlakes.bin.infer_wan_video \
        --config ${dir}/train.yaml \
        --checkpoint ${dir}/${checkpoint}.pt \
        --test_data /nfs-speech-cfs/wangzhou/s2s/vibehead/data/hdtf/test.jsonl \
        --result_dir outputs/s2_45000 \
        --num_steps 10 --cfg_scale ${scale}  --limit 1
    }
  done
}
done
