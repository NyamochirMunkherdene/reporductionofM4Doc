#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=3:00:00
#PJM -j
#PJM -o log_vary_toy_inference.out
#PJM -e log_vary_toy_inference.err
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

export checkpoint_dir="$base_dir/models/M4Doc_model/checkpoint-30000"

export nougat_model_dir="$base_dir/pretrained_models/nougat-small"

export dataset_dir="$base_dir/DoTA_dataset"

export output_dir="$base_dir/results/vary_toy"

export CONTAINER_IMAGE="/home/pj24001974/ku60000761/nlp-singularity/nlp-singularity.sif"

export ENV_FILE="$venv_dir/.env"

export vary_toy_source="$base_dir"


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


export CUDA_VISIBLE_DEVICES=0


mkdir -p "$output_dir"


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
            --num_processes 1 \
            --num_machines 1 \
            codes/inference.py \
            --checkpoint_dir \"$checkpoint_dir\" \
            --base_dir \"$base_dir\" \
            --trans_model_dir \"$trans_model_dir\" \
            --nougat_model_dir \"$nougat_model_dir\" \
            --dataset_dir \"$dataset_dir\" \
            --output_dir \"$output_dir\"
    "