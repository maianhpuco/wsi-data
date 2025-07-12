import os

# Datasets to process
datasets = ["kich", "kirp", "kirc", "luad", "lusc", "camelyon16"]
magnifications = ["5x", "10x"]

# Compute nodes to assign (compute-0-1 to compute-0-5)
nodes = [f"compute-0-{i}" for i in range(1, 6)]

# Directories and script path
sbatch_dir = "sbatch_scripts_clip"
log_dir = "logs_clip"

# Create output directories if they don't exist
os.makedirs(sbatch_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Determine script path based on dataset
def get_script_path(dataset):
    if dataset == "camelyon16":
        return "pipeline_fewshot/camelyon16/extract_clip_fp.py"
    else:
        return "pipeline_fewshot/tcga/extract_clip_fp.py"

# Template for SBATCH script
def create_sbatch_script(script_path, config_path, job_name, log_file, node, magnification):
    return f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={log_file}
#SBATCH --nodelist={node}
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "Starting {job_name} job on {node}"

python {script_path} --config {config_path} --magnification {magnification} --patch_size 256

echo "Done"
"""

# Generate SBATCH files
for idx, dataset in enumerate(datasets):
    for mag in magnifications:
        job_name = f"ecl{mag}_{dataset}"
        log_name = f"extract_clip_fp_{mag}_{dataset}.log"
        sbatch_name = f"extract_clip_fp_{mag}_{dataset}.sbatch"

        config_path = f"configs_maui/data_{dataset}.yaml"
        script_path = get_script_path(dataset)
        log_file = os.path.join(log_dir, log_name)
        node = nodes[idx % len(nodes)]

        sbatch_content = create_sbatch_script(script_path, config_path, job_name, log_file, node, mag)

        sbatch_path = os.path.join(sbatch_dir, sbatch_name)
        with open(sbatch_path, "w") as f:
            f.write(sbatch_content)

        print(f"--> Generated SBATCH: {sbatch_path} (Log: {log_file}, Node: {node})")
