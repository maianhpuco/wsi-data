#!/bin/bash
#SBATCH --job-name=pp_kirc
#SBATCH --output=logs/pp_kirc.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "Starting KIRC patching"
python pipeline_cls/tcga/create_patches_fp.py --config configs_maui/data_kirc.yaml
echo "Done"

