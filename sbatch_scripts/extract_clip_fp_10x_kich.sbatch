#!/bin/bash

#SBATCH --job-name=ecl10x_kich
#SBATCH --output=logs/extract_clip_fp_10x_kich.log
#SBATCH --nodelist=compute-0-1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting ecl10x_kich job on compute-0-1"

python pipeline_fewshot/tcga/extract_clip_fp.py --config configs_maui/data_kich.yaml --magnification 10x --patch_size 256

echo "Done"
