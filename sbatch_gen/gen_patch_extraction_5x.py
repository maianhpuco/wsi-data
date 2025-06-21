import os

# Datasets to process
datasets = ["kich", "kirp", "kirc", "luad", "lusc"]

# Compute nodes to assign (compute-0-1 to compute-0-5)
nodes = [f"compute-0-{i}" for i in range(1, 6)]

# Paths
script_path = "pipeline_fewshot/tcga/patch_extraction.py"
sbatch_dir = "sbatch_scripts"
log_dir = "logs"

# Create output directories if they don't exist
os.makedirs(sbatch_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Template for each SBATCH script
def create_sbatch_script(config_path, job_name, log_file, node):
    return f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={log_file}
#SBATCH --nodelist={node}5
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting {job_name} job on {node}"

python {script_path} --config {config_path} --patch_size 256 --magnification 5x   

echo "Done"
"""

# Generate SBATCH files
for i, dataset in enumerate(datasets):
    config_path = f"configs_maui/data_{dataset}.yaml"
    job_name = f"pextract_5x_{dataset}"
    log_file = os.path.join(log_dir, f"{job_name}.log")
    node = nodes[i % len(nodes)]  # Assign node in round-robin
    sbatch_content = create_sbatch_script(config_path, job_name, log_file, node)

    sbatch_path = os.path.join(sbatch_dir, f"{job_name}.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch_content)

    print(f"--> Generated SBATCH: {sbatch_path} (Node: {node})")
