#!/bin/bash

#SBATCH --job-name=pg_kirp
#SBATCH --output=logs/pg_kirp.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "Starting pg_kirp job"
python pipeline_fewshot/tcga/patch_generation.py --config configs_maui/data_kirp.yaml
echo "Done"
