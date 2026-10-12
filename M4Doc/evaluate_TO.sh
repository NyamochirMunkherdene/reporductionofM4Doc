#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -j
#PJM -o log_evaluate_100k.out
#PJM -e log_evaluate_100k.err
#PJM -L jobenv=singularity
#PJM -m b
#PJM -m e

set -euxo pipefail

module load singularity-ce/4.1.3


export project_dir=/home/pj24001974/ku60000761/project/test_project
export base_dir="$project_dir/M4Doc"
export venv_dir="$project_dir/Vary/.venv"

export zh_mmd_dir="$base_dir/DoTA_dataset/zh_mmd"
export split_json_file_path="$base_dir/DoTA_dataset/split_dataset.json"

export result_dir="$base_dir/results/base_model_DoTA/checkpoint-100kv1"
CONTAINER_IMAGE=/home/pj24001974/ku60000761/nlp-singularity/nlp-singularity.sif

cd "$base_dir"


singularity exec \
    --bind "$venv_dir:/opt/venv" \
    --bind "$project_dir:$project_dir" \
    "$CONTAINER_IMAGE" \
    bash -c "
        set -euxo pipefail

        echo '--- Activating virtual environment ---'
        source /opt/venv/bin/activate

        echo '--- Starting evaluation ---'

        python codes/evaluate.py \
            --split_json_file_path '$split_json_file_path' \
            --result_dir '$result_dir' \
            --zh_mmd_dir '$zh_mmd_dir'

        echo '--- Evaluation finished ---'
    "