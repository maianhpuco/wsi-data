#!/bin/bash

#SBATCH --job-name=pp_kich
#SBATCH --output=logs/pp_kich.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "Starting make job"
python pipeline_cls/tcga/create_patches_fp.py --config configs_maui/data_kich.yaml 
echo "Done"
