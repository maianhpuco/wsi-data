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

def copy_if_exists(src_file, dest_file, file_type=""):
    if os.path.exists(src_file):
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy2(src_file, dest_file)
        print(f"[✓] Copied {file_type}: {os.path.basename(src_file)}")
    else:
        print(f"[✗] Missing {file_type}: {os.path.basename(src_file)}")

def main(args, config): 
    dataset_name = config['dataset_name']
    print(f"Dataset: {dataset_name}")

    # === Source folders ===
    patch_dir = f"/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/{dataset_name}/png_patches/patch_256x256_10x"
    h5_dir = f"/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/{dataset_name}/features_fp/h5_files"
    pt_dir = f"/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/{dataset_name}/features_fp/pt_files"

    # === Destination folders ===
    dest_png_folder = f"/project/hnguyen2/mvu9/sample_data/{dataset_name}/png_patches"
    dest_h5_folder = f"/project/hnguyen2/mvu9/sample_data/{dataset_name}/features_fp/h5_files"
    dest_pt_folder = f"/project/hnguyen2/mvu9/sample_data/{dataset_name}/features_fp/pt_files"

    os.makedirs(dest_png_folder, exist_ok=True)
    os.makedirs(dest_h5_folder, exist_ok=True)
    os.makedirs(dest_pt_folder, exist_ok=True)

    # === Get all patch folder paths ===
    patch_folders = glob.glob(os.path.join(patch_dir, "*"))
    print("Total patch folders found:", len(patch_folders))

    # === Random sampling ===
    NUM_SAMPLES = 3
    sampled_folders = random.sample(patch_folders, NUM_SAMPLES)
    
    for src_folder in sampled_folders:
        slide_name = os.path.basename(src_folder)
        print(f"\n[Sample] Slide: {slide_name}")

        # === Copy PNG patch folder ===
        dest_patch = os.path.join(dest_png_folder, slide_name)
        if os.path.exists(dest_patch):
            shutil.rmtree(dest_patch)
        shutil.copytree(src_folder, dest_patch)
        print(f"[✓] Copied image folder: {slide_name}")

        # === Copy corresponding .h5 and .pt files ===
        h5_file_name = f"{slide_name}.h5"
        pt_file_name = f"{slide_name}.pt"

        src_h5_path = os.path.join(h5_dir, h5_file_name)
        dest_h5_path = os.path.join(dest_h5_folder, h5_file_name)
        copy_if_exists(src_h5_path, dest_h5_path, file_type=".h5")

        src_pt_path = os.path.join(pt_dir, pt_file_name)
        dest_pt_path = os.path.join(dest_pt_folder, pt_file_name)
        copy_if_exists(src_pt_path, dest_pt_path, file_type=".pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    main(args, config)
