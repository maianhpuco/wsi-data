import os

# Datasets you want to run patch generation for
datasets = ["kich", "kirp", "kirc", "luad", "lusc"]

# Paths
script_path = "pipeline_fewshot/tcga/patch_generation.py"
sbatch_dir = "sbatch_scripts"
log_dir = "logs"

# Create output directories if they don't exist
os.makedirs(sbatch_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Template for SBATCH script
def create_sbatch_script(config_path, job_name, log_file):
    return f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={log_file}
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=5-00:00:00  # 5 days (format: D-HH:MM:SS)
 
 
echo "Starting {job_name} job"

python {script_path} --config {config_path} --patch_size 256 --magnification 10x  
echo "Done"
"""

# Generate scripts
for dataset in datasets:
    config_path = f"configs_maui/data_{dataset}.yaml"
    job_name = f"pg_{dataset}"
    log_file = os.path.join(log_dir, f"{job_name}.log")
    sbatch_content = create_sbatch_script(config_path, job_name, log_file)

    sbatch_path = os.path.join(sbatch_dir, f"{job_name}.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch_content)

    print(f"--> Generated SBATCH: {sbatch_path}")
