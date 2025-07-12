#!/bin/bash

#SBATCH --job-name=create_patches_fp_10x_kirc
#SBATCH --output=logs/create_patches_fp_10x_kirc.log
#SBATCH --nodelist=compute-0-3
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting create_patches_fp_10x_kirc job on compute-0-3"

python pipeline_fewshot/tcga/create_patches_fp_10x.py --config configs_maui/data_kirc.yaml

echo "Done"
