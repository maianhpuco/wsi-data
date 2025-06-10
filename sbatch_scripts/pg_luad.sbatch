#!/bin/bash

#SBATCH --job-name=pg_luad
#SBATCH --output=logs/pg_luad.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 5 days (format: D-HH:MM:SS)
 
 
echo "Starting pg_luad job"

python pipeline_fewshot/tcga/sampling_for_local_test.py --config configs_maui/data_luad.yaml 
echo "Done"
