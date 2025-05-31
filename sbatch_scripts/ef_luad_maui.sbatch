#!/bin/bash

#SBATCH --job-name=ef_luad
#SBATCH --output=logs/ef_luad.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

echo "Starting feature extraction for LUAD"
python pipeline_cls/tcga/extract_features_fp.py --config configs_maui/data_luad.yaml
echo "Done"
