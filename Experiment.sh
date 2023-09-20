#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python Train_Codec.py \
--model_name SCR \
--exp_name test \
--train_config Custom_v1 \
