#!/bin/bash

#SBATCH --job-name=pp_kich
#SBATCH --output=sbatches/logs/pp_kich.out       
#SBATCH --error=sbatches/logs/pp_kich.err        # Save stderr to logs/ with job ID
#SBATCH --gres=gpu:1                       # If using GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

echo "Starting patch extraction for KICH..."
python create_patches_fp.py --config configs/data_kich.yaml
echo "Patch extraction complete."
