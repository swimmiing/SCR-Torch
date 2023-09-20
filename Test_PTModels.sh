#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python Test_Codec.py \
--model_name SCR \
--exp_name released \
--quality_level 4.2 \
--epochs None \
--data_path {put dataset directory} \
--save_path {put logging directory}