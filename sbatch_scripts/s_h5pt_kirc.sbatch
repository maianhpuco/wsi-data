#!/bin/bash

#SBATCH --job-name=s_h5pt_kirc
#SBATCH --output=logs/s_h5pt_kirc.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting s_h5pt_kirc job"

python pipeline_fewshot/tcga/sampling_copy_h5_and_pt.py --config configs_maui/data_kirc.yaml

echo "Done"
