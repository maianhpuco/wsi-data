#!/bin/bash

#SBATCH --job-name=pp_camelyon16
#SBATCH --output=logs/pp_camelyon16.log
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4

echo "Starting make job"
make pp_camelyon16_maui
echo "Done"
