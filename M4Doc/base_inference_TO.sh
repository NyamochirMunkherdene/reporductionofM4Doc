#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -o log_for_refrence.out
#PJM -e log_for_refrence.err
#PJM -L jobenv=singularity
#PJM -m b
#PJM -m e

set -euxo pipefail

module load gcc cuda cudnn nccl hpcx
module load singularity-ce/4.1.3

export project_dir=/home/pj24001974/ku60000761/project/test_project
export base_dir="$project_dir/M4Doc"
export venv_dir="$project_dir/Vary/.venv"

export trans_model_dir="$base_dir/models/dota_trans_jieba_lm_1536_v2/checkpoint-74000"

export en_tokenizer_dir="$base_dir/utils/en_tokenizer"
export zh_tokenizer_dir="$base_dir/utils/zh_tokenizer"

export en_mmd_dir="$base_dir/DoTA_dataset/en_mmd"
export split_json_file_path="$base_dir/DoTA_dataset/split_dataset.json"

export output_dir="$base_dir/results/predictions/dota_trans_jieba_lm_1536_v2/checkpoint-48000"

export CONTAINER_IMAGE=/home/pj24001974/ku60000761/nlp-singularity/nlp-singularity.sif

cd "$base_dir"

ENV_FILE="$venv_dir/.env"

if [ -f "$ENV_FILE" ]; then
    set +x
    set -a
    source "$ENV_FILE"
    set +a
    set -x
else
    echo "ERROR: Environment file not found: $ENV_FILE"
    exit 1
fi

mkdir -p "$output_dir"

singularity exec \
    --nv \
    --bind "$venv_dir:/opt/venv" \
    --bind "$project_dir:$project_dir" \
    --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    "$CONTAINER_IMAGE" \
    /opt/venv/bin/python \
    "$base_dir/codes/inference_TO.py" \
    --model_dir "$trans_model_dir" \
    --en_tokenizer_dir "$en_tokenizer_dir" \
    --zh_tokenizer_dir "$zh_tokenizer_dir" \
    --en_mmd_dir "$en_mmd_dir" \
    --split_json_file_path "$split_json_file_path" \
    --output_dir "$output_dir" \
    --split_key test_name_list \
    --max_length 1536 \
    --num_beams 4