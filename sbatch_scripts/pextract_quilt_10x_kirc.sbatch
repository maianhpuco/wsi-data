#!/bin/bash

#SBATCH --job-name=pextract_quilt_10x_kirc
#SBATCH --output=logs/pextract_quilt_10x_kirc.log
#SBATCH --nodelist=compute-0-3
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting pextract_quilt_10x_kirc job on compute-0-3"

python pipeline_fewshot/tcga/patch_extraction_quilt.py --config configs_maui/data_kirc.yaml --patch_size 256 --magnification 10x   

echo "Done"
