#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=4
#PJM -L elapse=24:00:00
#PJM -j
#PJM -o log_base_train_100k.out
#PJM -e log_base_train_100k.err
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
export nougat_model_dir="$base_dir/pretrained_models/nougat-small"

export dataset_dir="$base_dir/DoTA_dataset"
export output_dir="$base_dir/models/Base_model"

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
    bash -c "
        set -euxo pipefail

        source /opt/venv/bin/activate

        echo \"Python: \$(which python)\"
        echo \"Accelerate: \$(which accelerate)\"
        python -c \"import accelerate; print(accelerate.__version__)\"

        accelerate launch \
            --num_processes 4 \
            --num_machines 1 \
            \"$base_dir/codes/base_train.py\" \
            --base_dir \"$base_dir\" \
            --trans_model_dir \"$trans_model_dir\" \
            --nougat_model_dir \"$nougat_model_dir\" \
            --dataset_dir \"$dataset_dir\" \
            --output_dir \"$output_dir/100k\" \
            --max_length 1536 \
            --max_steps 100000 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --learning_rate 5e-5 \
            --weight_decay 0.01 \
            --warmup_steps 1000 \
            --logging_steps 10 \
            --eval_steps 250 \
            --save_steps 250 \
            --save_total_limit 3 \
            --gradient_checkpointing \
            --wandb_project \"M4Doc-Base-DIMT\" \
            --wandb_run_name \"base-dimt-10k-4gpu\" \
            --bf16 \
            --tf32
    "
