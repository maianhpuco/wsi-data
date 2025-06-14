#!/bin/bash

#SBATCH --job-name=s_lusc
#SBATCH --output=logs/s_lusc.log
#SBATCH --nodelist=compute-0-5
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting s_lusc job on compute-0-5"

python pipeline_fewshot/tcga/sampling_for_local_test.py --config configs_maui/data_lusc.yaml

echo "Done"
