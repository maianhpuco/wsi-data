#!/bin/bash

#SBATCH --job-name=s_h5pt_kich
#SBATCH --output=logs/s_h5pt_kich.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting s_h5pt_kich job"

python pipeline_fewshot/tcga/sampling_copy_h5_and_pt.py --config configs_maui/data_kich.yaml

echo "Done"
