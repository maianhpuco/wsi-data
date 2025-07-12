#!/bin/bash

#SBATCH --job-name=cpfp_10x_luad
#SBATCH --output=logs/create_patches_fp_10x_luad.log
#SBATCH --nodelist=compute-0-4
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting cpfp_10x_luad job on compute-0-4"

python pipeline_fewshot/tcga/create_patches_fp_10x.py --config configs_maui/data_luad.yaml

echo "Done"
