#!/bin/bash

. ./path.sh || exit 1;

export OMP_NUM_THREADS=1

# export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # bond0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0  # 1 金山机器
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_NET_GDR_LEVEL=2
export NCCL_PXN_DISABLE=0

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-""}
if [ -z "$cuda_visible_devices" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Using default device_ids."
  device_ids=(0 1 2 3 4 5 6 7)
else
  IFS=',' read -r -a device_ids <<< "$cuda_visible_devices"
  echo "Using CUDA_VISIBLE_DEVICES: $cuda_visible_devices"
fi
echo "Parsed device_ids: ${device_ids[@]}"

stage=3
stop_stage=3

num_nodes=$1
rank=$2
train_set="zh"
echo ${num_nodes} ${rank}

dir=exp/s5_1p7_all_semantic
tensorboard_dir=${dir}/tensorboard
num_workers=4
prefetch=2

train_engine=torch_ddp # torch_ddp deepspeed
train_config=conf/run_stage1.yaml


. tools/parse_options.sh || exit 1;

set -u
set -o pipefail

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # 多机 latent 预提取。每台机跑一次本脚本, 传 num_nodes / node_rank = $1 $2
  # (和 stage3 训练同约定)。全局分片: shard = node_rank * gpus_per_node + local_gpu,
  # world = num_nodes * gpus_per_node, 每张卡拿唯一 shard, extract_latents.py 内按
  # lines[shard::world] 切分。断点续跑靠目标 .ptz 是否已存在。
  # 固定用 flashhead 环境的 python (依赖齐全: torch 2.7.1+cu126 + einops/decord),
  # 不依赖 shell 当前激活的环境。提取脚本已删掉 torchaudio, 故无需再 unset LD_LIBRARY_PATH。
  PYEXE=/data/joe/anaconda3/envs/vibehead/bin/python
  OUT=/nfs-speech-cfs/wangzhou/data/tts/VividHead/wan_latents
  mkdir -p exp/extract_log $OUT

  IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
  gpus_per_node=${#_gpus[@]}
  world=$(( num_nodes * gpus_per_node ))
  echo "extract: num_nodes=$num_nodes node_rank=$rank gpus_per_node=$gpus_per_node world=$world"

  for local_gpu in "${!_gpus[@]}"; do
    dev=${_gpus[$local_gpu]}
    shard=$(( rank * gpus_per_node + local_gpu ))
    CUDA_VISIBLE_DEVICES=$dev $PYEXE -m twinlakes.bin.extract_latents \
        --config conf/run_stage1.yaml \
        --in_list data/vivi/train.list \
        --out_dir $OUT \
        --out_list data/vivi/train_latent.$shard.list \
        --rank $shard --world_size $world \
        > exp/extract_log/shard_$shard.log 2>&1 &
  done
  wait
  echo "extract: node $rank done. 全部机器跑完后合并: cat data/vivi/train_latent.*.list > data/vivi/train_latent.list"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start finetune"
  mkdir -p ${dir}/log
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  dist_backend="nccl"

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes \
           --nproc_per_node=$num_gpus \
           --node_rank=$rank \
           --master_addr="10.126.203.172" \
           --master_port=54322 \
    twinlakes/bin/train.py \
      --config $train_config \
      --data_type "raw" \
      --train_data data/talker/train_talker_vivid.json  \
      --cv_data data/talker/dev.json  \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory > ${dir}/log/${rank}.log 2>&1 
fi

# data/vivi/train_liax.json
# data/talker/train.json

