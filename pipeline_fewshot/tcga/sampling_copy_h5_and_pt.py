import os
import argparse
import yaml
from tqdm import tqdm
import glob
import random
import shutil

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def copy_if_exists(src_file, dest_file):
    if os.path.exists(src_file):
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy2(src_file, dest_file)
        print(f"[✓] Copied: {os.path.basename(src_file)}")
    else:
        print(f"[✗] Missing: {src_file}")

def main(args, config): 
    dataset_name = config['dataset_name']
    print(f"Dataset: {dataset_name}")

    # === Source folders ===
    h5_dir = f"/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/{dataset_name}/features_fp/h5_files"
    pt_dir = f"/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/{dataset_name}/features_fp/pt_files"

    # === Destination folders ===
    dest_png_folder = f"/project/hnguyen2/mvu9/sample_data/{dataset_name}/png_patches"
    dest_h5_folder = f"/project/hnguyen2/mvu9/sample_data/{dataset_name}/features_fp/h5_files"
    dest_pt_folder = f"/project/hnguyen2/mvu9/sample_data/{dataset_name}/features_fp/pt_files"

    os.makedirs(dest_h5_folder, exist_ok=True)
    os.makedirs(dest_pt_folder, exist_ok=True)

    # === Get list of slide folders already copied into dest_png_folder ===
    existing_slides = sorted([
        f for f in os.listdir(dest_png_folder) 
        if os.path.isdir(os.path.join(dest_png_folder, f))
    ])

    print(f"Found {len(existing_slides)} existing slides in {dest_png_folder}")

    for slide_name in existing_slides:
        print(f"\n[Slide] {slide_name}")

        h5_file_name = f"{slide_name}.h5"
        pt_file_name = f"{slide_name}.pt"

        src_h5_path = os.path.join(h5_dir, h5_file_name)
        dest_h5_path = os.path.join(dest_h5_folder, h5_file_name)
        copy_if_exists(src_h5_path, dest_h5_path)

        src_pt_path = os.path.join(pt_dir, pt_file_name)
        dest_pt_path = os.path.join(dest_pt_folder, pt_file_name)
        copy_if_exists(src_pt_path, dest_pt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    main(args, config)
