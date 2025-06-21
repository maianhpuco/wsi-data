#!/bin/bash

#SBATCH --job-name=pextract_quilt_5x_kich
#SBATCH --output=logs/pextract_quilt_5x_kich.log
#SBATCH --nodelist=compute-0-1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting pextract_quilt_5x_kich job on compute-0-1"

python pipeline_fewshot/tcga/patch_extraction_quilt.py --config configs_maui/data_kich.yaml --patch_size 256 --magnification 5x   

echo "Done"
