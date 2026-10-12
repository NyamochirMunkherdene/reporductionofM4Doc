#!/bin/bash

# This script is used to do inference and generate the Chinese texts for the input images.
base_dir="/home/pj24001974/ku60000761/project/test_project/M4Doc"

trans_model_dir="$base_dir/models/dota_trans_jieba_lm_1536_v2/checkpoint-74000"

checkpoint_dir="$base_dir/models/M4Doc_model/checkpoint-30000"

nougat_model_dir="$base_dir/pretrained_models/nougat-small"

dataset_dir="$base_dir/DoTA_dataset"

output_dir="$base_dir/results"

export CUDA_VISIBLE_DEVICES=0

python codes/inference.py \
    --checkpoint_dir $checkpoint_dir \
    --base_dir $base_dir \
    --trans_model_dir $trans_model_dir \
    --nougat_model_dir $nougat_model_dir \
    --dataset_dir $dataset_dir \
    --output_dir $output_dir
