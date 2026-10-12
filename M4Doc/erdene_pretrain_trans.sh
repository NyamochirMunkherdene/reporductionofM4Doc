#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=2
#PJM -L elapse=24:00:00
#PJM -j
#PJM -o log.out
#PJM -e log.err
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

export trans_model_dir=$base_dir/models/trans_model

export en_tokenizer_dir=$base_dir/utils/en_tokenizer
export zh_tokenizer_dir=$base_dir/utils/zh_tokenizer

export en_mmd_dir=$base_dir/DoTA_dataset/en_mmd
export zh_mmd_dir=$base_dir/DoTA_dataset/zh_mmd

export split_json_file_path=$base_dir/DoTA_dataset/split_dataset.json

export deepspeed_config_file_path=$base_dir/utils/deepspeed_zero0.json

CONTAINER_IMAGE=/home/pj24001974/ku60000761/nlp-singularity/nlp-singularity.sif

ENV_FILE="$venv_dir/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "ERROR: Environment file not found: $ENV_FILE"
    exit 1
fi


singularity exec  \
	--nv \
	--bind "$venv_dir:/opt/venv" \
	--bind "$project_dir:$project_dir" \
	--env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
	--env "HF_TOKEN=${HF_TOKEN:-}" \
	${CONTAINER_IMAGE} \
  bash -c "echo '--- Activating virtual environment inside container ---' && \
            source /opt/venv/bin/activate && \
            echo '--- Executing Python script ---' && \
            accelerate launch \
              --num_processes 2 \
              --num_machines 1 codes/pretrain_trans.py \
              --en_tokenizer_dir $en_tokenizer_dir \
              --zh_tokenizer_dir $zh_tokenizer_dir \
              --en_mmd_dir $en_mmd_dir \
              --zh_mmd_dir $zh_mmd_dir \
              --split_json_file_path $split_json_file_path \
              --output_dir $trans_model_dir \
              --batch_size 64 \
              --batch_size_per_gpu 4 \
              --num_train_epochs 1 \
              --eval_steps 1000 \
              --save_steps 1000 \
              --dataloader_num_workers 8 \
              --deepspeed $deepspeed_config_file_path && \
              echo '--- Python script finished ---'"


