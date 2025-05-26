# pipeline_cls/tcga/pyramidal_processing.py

import os
import subprocess
import argparse
import yaml
from tqdm import tqdm

def convert_to_pyramidal(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tif_files = [f for f in os.listdir(input_dir) if f.endswith(".tif") or f.endswith(".tiff")]
    print(len(tif_files), "TIFF files found in", input_dir)
    for filename in tqdm(tif_files, desc=f"Converting {input_dir}"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        cmd = [
            "vips", "tiffsave", input_path, output_path,
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

    source = config["ori_source_dir"]
    target = config["source_dir"]

    convert_to_pyramidal(source, target)
