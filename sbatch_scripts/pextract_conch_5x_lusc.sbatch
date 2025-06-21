#!/bin/bash

#SBATCH --job-name=pextract_conch_5x_lusc
#SBATCH --output=logs/pextract_conch_5x_lusc.log
#SBATCH --nodelist=compute-0-5
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting pextract_conch_5x_lusc job on compute-0-5"

python pipeline_fewshot/tcga/patch_extraction_quilt.py --config configs_maui/data_lusc.yaml --patch_size 256 --magnification 5x   

echo "Done"
