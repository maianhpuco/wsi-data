#!/bin/bash

#SBATCH --job-name=pextract_kirp
#SBATCH --output=logs/pextract_kirp.log
#SBATCH --nodelist=compute-0-2
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting pextract_kirp job on compute-0-2"

python pipeline_fewshot/tcga/patch_extraction.py --config configs_maui/data_kirp.yaml

echo "Done"
