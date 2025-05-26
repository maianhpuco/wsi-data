# pipeline_cls/tcga/pyramidal_processing.py

import os
import sys
import subprocess
import argparse
import yaml
from tqdm import tqdm

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)

def find_svs_files_recursive(input_dir):
    svs_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".svs"):
                svs_files.append(os.path.join(root, f))
    return svs_files

def convert_to_pyramidal(input_dir, output_dir):
    svs_files = find_svs_files_recursive(input_dir)
    print(len(svs_files), "SVS files found in", input_dir)
    
    for svs_path in tqdm(svs_files, desc=f"Converting {input_dir}"):
        # Extract relative path from UUID subfolder
        rel_path = os.path.relpath(svs_path, start=input_dir)  # e.g., uuid-123/slide_001.svs
        rel_dir = os.path.dirname(rel_path)                    # e.g., uuid-123
        filename = os.path.basename(svs_path).replace(".svs", ".tiff")

        # Create output directory <target>/<uuid> if not exists
        target_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(target_dir, exist_ok=True)

        output_path = os.path.join(target_dir, filename)

        cmd = [
            "vips", "tiffsave", svs_path, output_path,
            "--tile", "--pyramid", "--compression", "deflate",
            "--tile-width", "512", "--tile-height", "512"
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert: {svs_path}\n{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    source = config["paths"]["ori_source_dir"]
    target = config["paths"]["source_dir"]

    convert_to_pyramidal(source, target)
