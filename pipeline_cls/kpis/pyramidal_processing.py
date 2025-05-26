import os
import subprocess
from pathlib import Path
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

# Define paths
source_dirs = {
    # "train": "/home/mvu9/processing_datasets/kidney_pathology_image/reorganized_rawdata/images/train", 
    # "test":  "/home/mvu9/processing_datasets/kidney_pathology_image/reorganized_rawdata/images/test", 
    "validation": "/home/mvu9/processing_datasets/kidney_pathology_image/reorganized_rawdata/images/validation", 
}
output_dirs = {
    # "train": "/home/mvu9/processing_datasets/kidney_pathology_image/reorganized_rawdata/pyramidal_images/train", 
    # "test":  "/home/mvu9/processing_datasets/kidney_pathology_image/reorganized_rawdata/pyramidal_images/test", 
    "validation": "/home/mvu9/processing_datasets/kidney_pathology_image/reorganized_rawdata/pyramidal_images/validation", 
}

# Run conversion
for split in ["train", "test", "validation"]:
    convert_to_pyramidal(source_dirs[split], output_dirs[split])
