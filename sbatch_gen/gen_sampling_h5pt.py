import os

# === List of datasets to generate SBATCH files for ===
datasets = ["kich", "kirp", "kirc", "luad", "lusc"]

# === Paths ===
script_path = "pipeline_fewshot/tcga/sampling_for_local_test.py"
sbatch_dir = "sbatch_scripts"
log_dir = "logs"

# === Ensure output directories exist ===
os.makedirs(sbatch_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# === SBATCH script template ===
def create_sbatch_script(config_path, job_name, log_file):
    return f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={log_file}
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting {job_name} job"

python {script_path} --config {config_path}

echo "Done"
"""

# === Generate SBATCH scripts ===
for dataset in datasets:
    config_path = f"configs_maui/data_{dataset}.yaml"
    job_name = f"s_h5pt_{dataset}"
    log_file = os.path.join(log_dir, f"{job_name}.log")
    sbatch_content = create_sbatch_script(config_path, job_name, log_file)

    sbatch_path = os.path.join(sbatch_dir, f"{job_name}.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch_content)

    print(f"[✓] Generated SBATCH script: {sbatch_path}")
