#!/bin/bash

#SBATCH --job-name=cpfp_5x_kirp
#SBATCH --output=logs/cpfp_5x_kirp.log
#SBATCH --nodelist=compute-0-2
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting cpfp_5x_kirp job on compute-0-2"

python pipeline_fewshot/tcga/create_patches_fp_5x.py --config configs_maui/data_kirp.yaml

echo "Done"
