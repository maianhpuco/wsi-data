#!/bin/bash

#SBATCH --job-name=ecl5x_lusc
#SBATCH --output=logs/extract_clip_fp_5x_lusc.log
#SBATCH --nodelist=compute-0-5
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting ecl5x_lusc job on compute-0-5"

python pipeline_fewshot/tcga/extract_clip_fp.py --config configs_maui/data_lusc.yaml --magnification 5x --patch_size 256

echo "Done"
