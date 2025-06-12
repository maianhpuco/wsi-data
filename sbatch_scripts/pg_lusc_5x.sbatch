#!/bin/bash

#SBATCH --job-name=pg_lusc_5x
#SBATCH --output=logs/pg_lusc_5x.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=5-00:00:00  # 5 days (format: D-HH:MM:SS)
 
 
echo "Starting pg_lusc_5x job"

python pipeline_fewshot/tcga/patch_generation.py --config configs_maui/data_lusc.yaml --patch_size 256 --magnification 5x  
echo "Done"
