#!/bin/bash

#SBATCH --job-name=cp10x_kich
#SBATCH --output=logs/create_patches_fp_10x_kich.log
#SBATCH --nodelist=compute-0-1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting cp10x_kich job on compute-0-1"

python pipeline_fewshot/tcga/create_patches_fp_10x.py --config configs_maui/data_kich.yaml

echo "Done"
