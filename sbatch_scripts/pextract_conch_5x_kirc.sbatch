#!/bin/bash

#SBATCH --job-name=pextract_conch_5x_kirc
#SBATCH --output=logs/pextract_conch_5x_kirc.log
#SBATCH --nodelist=compute-0-3
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting pextract_conch_5x_kirc job on compute-0-3"

python pipeline_fewshot/tcga/patch_extraction_conch.py --config configs_maui/data_kirc.yaml --patch_size 256 --magnification 5x   

echo "Done"
