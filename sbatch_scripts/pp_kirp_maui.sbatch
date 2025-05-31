#!/bin/bash
#SBATCH --job-name=pp_kirp
#SBATCH --output=logs/pp_kirp.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "Starting KIRP patching"
python pipeline_cls/tcga/create_patches_fp.py --config configs_maui/data_kirp.yaml
echo "Done"

