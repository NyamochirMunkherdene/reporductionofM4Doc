#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=06:00:00
#PJM -L jobenv=singularity
#PJM -j
#PJM -o base_inference.out
#PJM -e base_inference.err
#PJM -m b
#PJM -m e

set -euxo pipefail

module load gcc cuda cudnn nccl hpcx
module load singularity-ce/4.1.3

export project_dir=/home/pj24001974/ku60000761/project/test_project
export base_dir="$project_dir/M4Doc"
export venv_dir="$project_dir/Vary/.venv"

export trans_model_dir="$base_dir/models/trans_model_v3/checkpoint-24000"
export checkpoint_dir="$base_dir/models/Base_model/final_model"

# Replace this with the actual Nougat model directory.
export nougat_model_dir="$base_dir/pretrained_models/nougat-small"

export dataset_dir="$base_dir/DoTA_dataset"
export output_dir="$base_dir/results/base_model_DoTA/results_0805"

export CONTAINER_IMAGE=/home/pj24001974/ku60000761/nlp-singularity/nlp-singularity.sif

cd "$base_dir"
mkdir -p "$output_dir"

# Validate required files and directories.
test -f "$trans_model_dir/config.json"
test -f "$nougat_model_dir/config.json"
test -f "$checkpoint_dir/state_dict/nougat_encoder.bin"
test -f "$checkpoint_dir/state_dict/trans_decoder.bin"
test -f "$checkpoint_dir/state_dict/nougat_down_proj.bin"
test -f "$dataset_dir/split_dataset.json"
test -d "$dataset_dir/imgs"

singularity exec \
    --nv \
    --bind "$venv_dir:/opt/venv" \
    --bind "$project_dir:$project_dir" \
    "$CONTAINER_IMAGE" \
    bash -c "
        set -euo pipefail

        source /opt/venv/bin/activate
        cd '$base_dir'

        python '$base_dir/codes/ /base_inference_DOTA.py' \
            --checkpoint_dir '$checkpoint_dir' \
            --base_dir '$base_dir' \
            --trans_model_dir '$trans_model_dir' \
            --nougat_model_dir '$nougat_model_dir' \
            --dataset_dir '$dataset_dir' \
            --output_dir '$output_dir' \
    "

echo "Inference completed."