#!/bin/bash

#SBATCH --job-name=pg_luad
#SBATCH --output=logs/pg_luad.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "Starting pg_luad job"
python pipeline_fewshot/tcga/patch_generation.py --config configs_maui/data_luad.yaml
echo "Done"
