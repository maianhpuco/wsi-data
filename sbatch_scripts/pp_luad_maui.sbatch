#!/bin/bash
#SBATCH --job-name=pp_luad
#SBATCH --output=logs/pp_luad.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "Starting LUAD patching"
python pipeline_cls/tcga/create_patches_fp.py --config configs_maui/data_luad.yaml
echo "Done"

