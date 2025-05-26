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
    os.makedirs(output_dir, exist_ok=True)
    svs_files = find_svs_files_recursive(input_dir)
    print(len(svs_files), "SVS files found in", input_dir)
    for svs_path in tqdm(svs_files, desc=f"Converting {input_dir}"):
        filename = os.path.basename(svs_path)
        output_path = os.path.join(output_dir, filename.replace(".svs", ".tiff"))

        cmd = [
            "vips", "tiffsave", svs_path, output_path,
            "--tile", "--pyramid", "--compression", "deflate",
            "--tile-width", "512", "--tile-height", "512"
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert: {filename}\n{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    source = config["paths"]["ori_source_dir"]
    target = config["paths"]["source_dir"]

    convert_to_pyramidal(source, target)
