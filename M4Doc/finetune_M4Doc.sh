#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=4
#PJM -L elapse=24:00:00
#PJM -j
#PJM -o log_vary_toy_model.out
#PJM -e log_vary_toy_model.err
#PJM -L jobenv=singularity
#PJM -m b
#PJM -m e

set -euxo pipefail
module load gcc cuda cudnn nccl hpcx
module load singularity-ce/4.1.3

export project_dir=/home/pj24001974/ku60000761/project/test_project
export base_dir="$project_dir/M4Doc"
export venv_dir="$project_dir/Vary/.venv"

# Vary-toy SOURCE CODE
export vary_toy_source="$project_dir/Vary-toy/Vary-master"

# Vary-toy WEIGHTS
export mllm_model_dir="$project_dir/vary-toy-weights"

export trans_model_dir="$base_dir/models/dota_trans_jieba_lm_1536_v2/checkpoint-74000"
export nougat_model_dir="$base_dir/pretrained_models/nougat-small"
export dataset_dir="$base_dir/DoTA_dataset"
export m4doc_model_dir="$base_dir/models/M4Doc_model"

export deepspeed_config_file_path="$base_dir/utils/deepspeed_zero0.json"

export CONTAINER_IMAGE=/home/pj24001974/ku60000761/nlp-singularity/nlp-singularity.sif
export ENV_FILE="$venv_dir/.env"

cd "$base_dir"

if [ -f "$ENV_FILE" ]; then
    set +x
    set -a
    source "$ENV_FILE"
    set +a
    set -x
else
    echo "ERROR: Environment file not found: ENV_FILE"
    exit 1
fi
mkdir -p "$m4doc_model_dir"
export CUDA_VISIBLE_DEVICES=0,1,2,3

singularity exec \
    --nv \
    --bind "$venv_dir:/opt/venv" \
    --bind "$project_dir:$project_dir" \
    --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    --env "PYTHONPATH=$vary_toy_source" \
    "$CONTAINER_IMAGE" \
    bash -c "
        set -euxo pipefail

        source /opt/venv/bin/activate



        /opt/venv/bin/accelerate launch \
            --num_processes 4 \
            --num_machines 1 \
            codes/finetune_M4Doc_vary.py \
            --base_dir '$base_dir' \
            --mllm_model_dir '$mllm_model_dir' \
            --trans_model_dir '$trans_model_dir' \
            --nougat_model_dir '$nougat_model_dir' \
            --dataset_dir '$dataset_dir' \
            --output_dir '$m4doc_model_dir' \
            --batch_size 64 \
            --batch_size_per_gpu 2 \
            --dataloader_num_workers 16 \
            --deepspeed '$deepspeed_config_file_path'
    "