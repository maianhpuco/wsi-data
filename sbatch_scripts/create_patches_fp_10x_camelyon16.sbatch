#!/bin/bash

#SBATCH --job-name=cpfp_10x_camelyon16
#SBATCH --output=logs/create_patches_fp_10x_camelyon16.log
#SBATCH --nodelist=compute-0-6
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting create_patches_fp_10x_camelyon16 job on compute-0-6"

python pipeline_fewshot/camelyon/create_patches_fp_10x.py --config configs_maui/data_camelyon16.yaml

echo "Done"
