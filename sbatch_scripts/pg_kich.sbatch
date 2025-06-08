#!/bin/bash

#SBATCH --job-name=pg_kich
#SBATCH --output=logs/pg_kich.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "Starting pg_kich job"
python pipeline_fewshot/tcga/patch_generation.py --config configs_maui/data_kich.yaml --patch_size 256 --magnification 10x  
echo "Done"
