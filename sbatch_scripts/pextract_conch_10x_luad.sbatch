#!/bin/bash

#SBATCH --job-name=pextract_conch_10x_luad
#SBATCH --output=logs/pextract_conch_10x_luad.log
#SBATCH --nodelist=compute-0-4
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting pextract_conch_10x_luad job on compute-0-4"

python pipeline_fewshot/tcga/patch_extraction_conch.py --config configs_maui/data_luad.yaml --patch_size 256 --magnification 10x   

echo "Done"
