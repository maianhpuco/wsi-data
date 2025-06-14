#!/bin/bash

#SBATCH --job-name=pextract_kirc
#SBATCH --output=logs/pextract_kirc.log
#SBATCH --nodelist=compute-0-3
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting pextract_kirc job on compute-0-3"

python pipeline_fewshot/tcga/patch_extraction.py --config configs_maui/data_kirc.yaml

echo "Done"
