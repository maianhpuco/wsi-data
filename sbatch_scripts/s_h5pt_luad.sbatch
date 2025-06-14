#!/bin/bash

#SBATCH --job-name=s_h5pt_luad
#SBATCH --output=logs/s_h5pt_luad.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting s_h5pt_luad job"

python pipeline_fewshot/tcga/sampling_for_local_test.py --config configs_maui/data_luad.yaml

echo "Done"
