#!/bin/bash

#SBATCH --job-name=pp_camelyon16
#SBATCH --output=logs/pp_camelyon16.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "Starting make job"
python pipeline_cls/camelyon16/create_patches_fp.py --config configs_maui/data_camelyon16.yaml
echo "Done"
