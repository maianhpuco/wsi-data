#!/bin/bash

#SBATCH --job-name=ec10x_lusc
#SBATCH --output=logs/extract_conch_fp_10x_lusc.log
#SBATCH --nodelist=compute-0-5
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting ec10x_lusc job on compute-0-5"

python pipeline_fewshot/tcga/extract_conch_fp.py --config configs_maui/data_lusc.yaml --magnification 10x --patch_size 256

echo "Done"
