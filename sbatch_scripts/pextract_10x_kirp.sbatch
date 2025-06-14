#!/bin/bash

#SBATCH --job-name=pextract_10x_kirp
#SBATCH --output=logs/pextract_10x_kirp.log
#SBATCH --nodelist=compute-0-2
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting pextract_10x_kirp job on compute-0-2"

python pipeline_fewshot/tcga/patch_extraction.py --config configs_maui/data_kirp.yaml --patch_size 256 --magnification 10x   

echo "Done"
