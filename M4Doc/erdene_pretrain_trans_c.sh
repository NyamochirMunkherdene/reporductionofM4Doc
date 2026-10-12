#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=4
#PJM -L elapse=4:00:00
#PJM -j
#PJM -o log_debug.out
#PJM -e log_debug.err
#PJM -L jobenv=singularity
#PJM -m b
#PJM -m e

set -euxo pipefail

module load gcc cuda cudnn nccl hpcx
module load singularity-ce/4.1.3

cd /home/pj24001974/ku60000761/project/test_project/M4Doc

export project_dir=/home/pj24001974/ku60000761/project/test_project
export base_dir="$project_dir/M4Doc"
export venv_dir="$project_dir/Vary/.venv"

export pretrained_model_dir="$base_dir/models/un_trans_jieba_lm_1e-4/final_model"
export finetune_model_dir="$base_dir/models/dota_trans_jieba_lm_1536_v2"

export en_tokenizer_dir="$base_dir/utils/en_tokenizer"
export zh_tokenizer_dir="$base_dir/utils/zh_tokenizer"

export en_mmd_dir="$base_dir/DoTA_dataset/en_mmd"
export zh_mmd_dir="$base_dir/DoTA_dataset/zh_mmd"

export split_json_file_path="$base_dir/DoTA_dataset/split_dataset.json"

export deepspeed_config_file_path="$base_dir/utils/deepspeed_zero0.json"

CONTAINER_IMAGE=/home/pj24001974/ku60000761/nlp-singularity/nlp-singularity.sif

ENV_FILE="$venv_dir/.env"

if [ -f "$ENV_FILE" ]; then
    set +x
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "ERROR: Environment file not found: $ENV_FILE"
    exit 1
fi

singularity exec \
    --nv \
    --bind "$venv_dir:/opt/venv" \
    --bind "$project_dir:$project_dir" \
    --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    "$CONTAINER_IMAGE" \
    bash -c "
        echo '--- Starting DoTA text-only fine-tuning ---'&& \
            source /opt/venv/bin/activate && \
            echo '--- Executing Python script ---' && \
            accelerate launch \
                --num_processes 4 \
                --num_machines 1 codes/finetune_trans.py \
                --pretrained_model_dir $pretrained_model_dir \
                --en_tokenizer_dir $en_tokenizer_dir \
                --zh_tokenizer_dir $zh_tokenizer_dir \
                --en_mmd_dir $en_mmd_dir \
                --zh_mmd_dir $zh_mmd_dir \
                --split_json_file_path $split_json_file_path \
                --output_dir $finetune_model_dir \
                --max_length 1536 \
                --batch_size 64 \
                --batch_size_per_gpu 1 \
                --learning_rate 5e-5 \
                --num_train_epochs 40 \
                --warmup_ratio 0.05 \
                --eval_steps 1000 \
                --save_steps 1000 \
                --logging_steps 10 \
                --dataloader_num_workers 16 \
                --deepspeed $deepspeed_config_file_path

            echo '--- DoTA fine-tuning finished ---'
        "