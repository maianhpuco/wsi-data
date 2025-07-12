#!/bin/bash

#SBATCH --job-name=create_patches_fp_10x_kirp
#SBATCH --output=logs/create_patches_fp_10x_kirp.log
#SBATCH --nodelist=compute-0-2
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting create_patches_fp_10x_kirp job on compute-0-2"

python pipeline_fewshot/tcga/create_patches_fp_10x.py --config configs_maui/data_kirp.yaml

echo "Done"
