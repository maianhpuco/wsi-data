#!/bin/bash

#SBATCH --job-name=cp10x_lusc
#SBATCH --output=logs/create_patches_fp_10x_lusc.log
#SBATCH --nodelist=compute-0-5
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting cp10x_lusc job on compute-0-5"

python pipeline_fewshot/tcga/create_patches_fp_10x.py --config configs_maui/data_lusc.yaml

echo "Done"
