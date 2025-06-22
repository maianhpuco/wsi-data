#!/bin/bash

#SBATCH --job-name=pg_10x_lusc
#SBATCH --output=logs/pg_10x_lusc.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=5-00:00:00  # 5 days (format: D-HH:MM:SS)
 
 
echo "Starting pg_10x_lusc job"

python pipeline_fewshot/tcga/patch_generation.py --config configs_maui/data_lusc.yaml --patch_size 256 --magnification 10x  
echo "Done"
