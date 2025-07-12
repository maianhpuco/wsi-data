#!/bin/bash

#SBATCH --job-name=ec5x_camelyon16
#SBATCH --output=logs/extract_conch_fp_5x_camelyon16.log
#SBATCH --nodelist=compute-0-1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting ec5x_camelyon16 job on compute-0-1"

python pipeline_fewshot/camelyon16/extract_conch_fp.py --config configs_maui/data_camelyon16.yaml --magnification 5x --patch_size 256

echo "Done"
