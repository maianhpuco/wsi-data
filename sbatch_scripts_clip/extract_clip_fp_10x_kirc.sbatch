#!/bin/bash

#SBATCH --job-name=ecl10x_kirc
#SBATCH --output=logs_clip/extract_clip_fp_10x_kirc.log
#SBATCH --nodelist=compute-0-3
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting ecl10x_kirc job on compute-0-3"

python pipeline_fewshot/tcga/extract_clip_fp.py --config configs_maui/data_kirc.yaml --magnification 10x --patch_size 256

echo "Done"
