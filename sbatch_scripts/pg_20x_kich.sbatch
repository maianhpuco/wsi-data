#!/bin/bash

#SBATCH --job-name=pg_20x_kich
#SBATCH --output=logs/pg_20x_kich.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=5-00:00:00  # 5 days (format: D-HH:MM:SS)
 
20
echo "Starting pg_20x_kich job"

python pipeline_fewshot/tcga/patch_generation.py --config configs_maui/data_kich.yaml --patch_size 256 --magnification 20x  
echo "Done"
