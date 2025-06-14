#!/bin/bash

#SBATCH --job-name=pextract_kich
#SBATCH --output=logs/pextract_kich.log
#SBATCH --nodelist=compute-0-1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting pextract_kich job on compute-0-1"

python pipeline_fewshot/tcga/patch_extraction.py --config configs_maui/data_kich.yaml

echo "Done"
