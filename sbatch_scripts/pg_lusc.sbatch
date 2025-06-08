#!/bin/bash

#SBATCH --job-name=pg_lusc
#SBATCH --output=logs/pg_lusc.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "Starting pg_lusc job"
python pipeline_fewshot/tcga/patch_generation.py --config configs_maui/data_lusc.yaml
echo "Done"
