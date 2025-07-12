#!/bin/bash

#SBATCH --job-name=create_patches_fp_5x_luad
#SBATCH --output=logs/create_patches_fp_5x_luad.log
#SBATCH --nodelist=compute-0-4
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting create_patches_fp_5x_luad job on compute-0-4"

python pipeline_fewshot/tcga/create_patches_fp_5x.py --config configs_maui/data_luad.yaml

echo "Done"
