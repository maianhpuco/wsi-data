#!/bin/bash
#SBATCH --job-name=pp_lusc
#SBATCH --output=logs/pp_lusc.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "Starting LUSC patching"
python pipeline_cls/tcga/create_patches_fp.py --config configs_maui/data_lusc.yaml
echo "Done"

