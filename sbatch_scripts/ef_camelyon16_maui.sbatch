#!/bin/bash

#SBATCH --job-name=ef_camelyon16
#SBATCH --output=logs/ef_camelyon16.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

echo "Starting feature extraction for CAMELYON16"
python pipeline_cls/camelyon16/extract_features_fp.py --config configs_maui/data_camelyon16.yaml
echo "Done"
