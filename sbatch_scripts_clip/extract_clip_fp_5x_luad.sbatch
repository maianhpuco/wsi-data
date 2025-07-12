#!/bin/bash

#SBATCH --job-name=ecl5x_luad
#SBATCH --output=logs_clip/extract_clip_fp_5x_luad.log
#SBATCH --nodelist=compute-0-4
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting ecl5x_luad job on compute-0-4"

python pipeline_fewshot/tcga/extract_clip_fp.py --config configs_maui/data_luad.yaml --magnification 5x --patch_size 256

echo "Done"
