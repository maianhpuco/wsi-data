import os

# Dataset info: dataset_name → (subfolder under pipeline_cls, job name)
datasets = {
    "camelyon16": "camelyon16",
    "kich": "tcga",
    "kirc": "tcga",
    "kirp": "tcga",
    "luad": "tcga",
    "lusc": "tcga",
}

sbatch_template = """#!/bin/bash

#SBATCH --job-name=ef_{name}
#SBATCH --output=logs/ef_{name}.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

echo "Starting feature extraction for {NAME}"
python pipeline_cls/{folder}/extract_features_fp.py --config configs_maui/data_{name}.yaml
echo "Done"
"""

for name, folder in datasets.items():
    sbatch_path = f"sbatch_scripts/ef_{name}_maui.sbatch"
    content = sbatch_template.format(name=name, NAME=name.upper(), folder=folder)
    
    with open(sbatch_path, "w") as f:
        f.write(content)

print("All feature extraction sbatch scripts generated.")
