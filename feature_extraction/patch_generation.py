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
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Generate patches from H5 files.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    return parser.parse_args()

def generate_patch(patch_file_name, slide_folder, patch_folder, save_folder, svs2uuid, define_patch_size, scale, save_name):
    patch_path = os.path.join(patch_folder, patch_file_name)
    slide_path = os.path.join(slide_folder, svs2uuid[patch_file_name.replace('.h5', '.svs')])

    f = h5py.File(patch_path, 'r')
    coords = f['coords']
    patch_level = coords.attrs['patch_level']
    patch_size = coords.attrs['patch_size']
    slide = openslide.OpenSlide(slide_path)
    try:
        magnification = int(float(slide.properties['aperio.AppMag']))
    except:
        magnification = 40
    save_path = os.path.join(save_folder, patch_file_name.replace('.h5', ''))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        if magnification == 40:
            resized_patch_size = int(patch_size / scale)
        elif magnification == 20:
            resized_patch_size = int(patch_size / (scale / 2))
        for coord in tqdm(coords, desc=f"Processing {patch_file_name}"):
            coord = coord.astype(np.int_)
            patch = slide.read_region(coord, int(patch_level), (int(patch_size), int(patch_size))).convert('RGB')
            patch = patch.resize((resized_patch_size, resized_patch_size))
            patch_name = f"{coord[0]}_{coord[1]}.png"
            patch_save_path = os.path.join(save_path, patch_name)
            patch.save(patch_save_path)
    else:
        print(f"{patch_file_name}: has been processed!")

def main():
    # Parse command-line arguments
    args = parse_args()
    config = load_config(args.config)

    # Extract paths from config
    slide_folder = config['paths']['source_dir']
    patch_folder = config['paths']['patch_h5_dir']
    save_folder = config['paths']['patch_png_dir']
    uuid_file = config['paths']['uuid_name_file']

    # Extract processing parameters
    define_patch_size = config['processing']['patch_size']
    patch_level = config['processing']['patch_level']

    # Set scale and save_name based on patch_size
    if define_patch_size == 256:
        scale = 1  # Adjust based on your magnification logic
        save_name = '40x'  # Adjust based on your desired naming
    else:
        raise ValueError(f"Unsupported patch_size: {define_patch_size}")

    # Create save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load UUID mapping
    uuid_data = pd.read_excel(uuid_file, engine='openpyxl', header=None)
    svs2uuid = {row[1].rstrip('\n'): row[0] for row in uuid_data.values}

    # Process patches using ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=16)
    all_file_names = [f for f in os.listdir(patch_folder) if f.endswith('.h5')]
    for patch_file_name in all_file_names:
        pool.submit(
            generate_patch,
            patch_file_name,
            slide_folder,
            patch_folder,
            save_folder,
            svs2uuid,
            define_patch_size,
            scale,
            save_name
        )
    pool.shutdown(wait=True)

if __name__ == "__main__":
    main()