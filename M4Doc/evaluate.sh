#!/bin/bash

# This script is used to calcaute BLEU, BLEU-PT and STEDS.
export project_dir=/home/pj24001974/ku60000761/project/test_project
export base_dir="$project_dir/M4Doc"
export venv_dir="$project_dir/Vary/.venv"

export zh_mmd_dir="$base_dir/DoTA_dataset/zh_mmd"
export split_json_file_path="$base_dir/DoTA_dataset/split_dataset.json"

export result_dir="$base_dir/results/base_model_DoTA/results_0806"

python codes/evaluate.py \
    --split_json_file_path "$split_json_file_path" \
    --result_dir "$result_dir" \
    --zh_mmd_dir "$zh_mmd_dir"
