import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def convert_to_pyramidal(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tif_files = [f for f in os.listdir(input_dir) if f.endswith(".tif") or f.endswith(".tiff")]

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

# Define paths
source_dirs = {
    "train": "/home/mvu9/datasets/glomeruli/train",
    "test": "/home/mvu9/datasets/glomeruli/test"
}
output_dirs = {
    "train": "/home/mvu9/processing_datasets/glomeruli_pyramidal/train",
    "test": "/home/mvu9/processing_datasets/glomeruli_pyramidal/test"
}

# Run conversion
for split in ["train", "test"]:
    convert_to_pyramidal(source_dirs[split], output_dirs[split])
