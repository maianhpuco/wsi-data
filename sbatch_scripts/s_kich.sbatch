#!/bin/bash

#SBATCH --job-name=s_kich
#SBATCH --output=logs/s_kich.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 5 days (format: D-HH:MM:SS)
 
 
echo "Starting s_kich job"

python pipeline_fewshot/tcga/sampling_for_local_test.py --config configs_maui/data_kich.yaml 
echo "Done"
