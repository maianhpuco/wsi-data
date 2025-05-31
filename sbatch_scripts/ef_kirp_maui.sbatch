#!/bin/bash

#SBATCH --job-name=ef_kirp
#SBATCH --output=logs/ef_kirp.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

echo "Starting feature extraction for KIRP"
python pipeline_cls/tcga/extract_features_fp.py --config configs_maui/data_kirp.yaml
echo "Done"
