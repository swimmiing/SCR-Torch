#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS="4"

python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 12345 \
Train_Codec.py \
--model_name SCR \
--exp_name test \
--train_config Custom_v1 \
--data_path {put dataset directory} \
--save_path {put logging directory}