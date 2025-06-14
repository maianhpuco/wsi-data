#!/bin/bash

#SBATCH --job-name=s_luad
#SBATCH --output=logs/s_luad.log
#SBATCH --nodelist=compute-0-4
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting s_luad job on compute-0-4"

python pipeline_fewshot/tcga/sampling_for_local_test.py --config configs_maui/data_luad.yaml

echo "Done"
