import os
import h5py
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import yaml
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate patches from H5 files using config.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    return parser.parse_args()

def generate_patch(patch_file_name, slide_folder, patch_folder, save_folder, svs2uuid, scale):
    try:
        patch_path = os.path.join(patch_folder, patch_file_name)
        key = patch_file_name.replace('.h5', '.svs')

        # Get UUID folder and find .svs inside it
        uuid_folder = os.path.join(slide_folder, svs2uuid[key])
        svs_files = [f for f in os.listdir(uuid_folder) if f.endswith('.svs')]
        if not svs_files:
            print(f"[ERROR] No .svs file found in {uuid_folder}")
            return
        slide_path = os.path.join(uuid_folder, svs_files[0])

        f = h5py.File(patch_path, 'r')
        coords = f['coords']
        patch_level = coords.attrs['patch_level']
        patch_size = coords.attrs['patch_size']

        slide = openslide.OpenSlide(slide_path)
        try:
            magnification = int(float(slide.properties.get('aperio.AppMag', 40)))
        except:
            magnification = 40

        save_path = os.path.join(save_folder, patch_file_name.replace('.h5', ''))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            resized_patch_size = int(patch_size / scale) if magnification == 40 else int(patch_size / (scale / 2))
            for coord in tqdm(coords, desc=patch_file_name, leave=False):
                coord = coord.astype(np.int_)
                patch = slide.read_region(coord, int(patch_level), (patch_size, patch_size)).convert('RGB')
                patch = patch.resize((resized_patch_size, resized_patch_size))
                patch_name = f"{coord[0]}_{coord[1]}.png"
                patch.save(os.path.join(save_path, patch_name))
        else:
            print(f"[INFO] {patch_file_name} has already been processed.")
    except Exception as e:
        print(f"[ERROR] Failed on {patch_file_name}: {e}")

def main():
    args = parse_args()
    config = load_config(args.config)

    slide_folder = config['paths']['source_dir']
    patch_folder = config['paths']['patch_h5_dir']
    save_folder = config['paths']['patch_png_dir']
    uuid_file = config['paths']['uuid_name_file']
    patch_size = config['processing']['patch_size']

    # Set scale
    if patch_size == 2048:
        scale = 8
    elif patch_size == 1024:
        scale = 4
    elif patch_size == 512:
        scale = 2
    else:
        raise ValueError(f"Unsupported patch size: {patch_size}")

    # Ensure folders exist
    os.makedirs(save_folder, exist_ok=True)

    # Load UUID map
    uuid_data = np.array(pd.read_excel(uuid_file, engine='openpyxl', header=None))
    svs2uuid = {row[1].strip(): row[0] for row in uuid_data}

    # List H5 files
    if not os.path.exists(patch_folder):
        print(f"[ERROR] Patch folder does not exist: {patch_folder}")
        return
    all_file_names = [f for f in os.listdir(patch_folder) if f.endswith('.h5')]

    print(f"[INFO] Found {len(all_file_names)} patch files to process...")

    # Process with thread pool
    pool = ThreadPoolExecutor(max_workers=16)
    for patch_file_name in all_file_names:
        pool.submit(generate_patch, patch_file_name, slide_folder, patch_folder, save_folder, svs2uuid, scale)
    pool.shutdown(wait=True)
    print("[INFO] All done.")

if __name__ == "__main__":
    main()
