import os
import sys 
import h5py
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import yaml
import argparse
# Get the absolute path of the parent of the parent directory
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)
print("Search path:", base_path) 



def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Generate patches from H5 files.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    return parser.parse_args()

def generate_patch(patch_file_name, slide_folder, patch_folder, save_folder, svs2uuid, define_patch_size, scale):
    try:
        print(f"[START] {patch_file_name}")

        key = patch_file_name.replace('.h5', '.svs')
        if key not in svs2uuid:
            print(f"[SKIP] Key {key} not found in UUID mapping.")
            return

        patch_path = os.path.join(patch_folder, patch_file_name)
        slide_path = os.path.join(slide_folder, svs2uuid[key])

        if not os.path.exists(slide_path):
            print(f"[SKIP] Slide file does not exist: {slide_path}")
            return

        f = h5py.File(patch_path, 'r')
        coords = f['coords']
        patch_level = coords.attrs['patch_level']
        patch_size = coords.attrs['patch_size']
    
        try:
            slide = openslide.OpenSlide(slide_path)
        except Exception as e:
            print(f"[ERROR] Failed to open slide {slide_path}: {e}")
            return

        try:
            magnification = int(float(slide.properties.get('aperio.AppMag', 40)))
        except:
            magnification = 40

        save_path = os.path.join(save_folder, patch_file_name.replace('.h5', ''))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        resized_patch_size = int(patch_size / scale) if magnification == 40 else int(patch_size / (scale / 2))

        for coord in tqdm(coords, desc=f"Processing {patch_file_name}", leave=False):
            coord = coord.astype(np.int_)
            patch = slide.read_region(coord, int(patch_level), (int(patch_size), int(patch_size))).convert('RGB')
            patch = patch.resize((resized_patch_size, resized_patch_size))
            patch_name = f"{coord[0]}_{coord[1]}.png"
            patch_save_path = os.path.join(save_path, patch_name)
            patch.save(patch_save_path)

        print(f"[DONE] {patch_file_name}")
    except Exception as e:
        print(f"[ERROR] {patch_file_name}: {e}")

def handle_future(future):
    try:
        future.result()
    except Exception as e:
        print(f"[THREAD ERROR]: {e}")

def main():
    args = parse_args()
    config = load_config(args.config)
    print("[INFO] Loaded config.")

    # Paths
    slide_folder = config['paths']['source_dir']
    patch_folder = config['paths']['patch_h5_dir']
    save_folder = config['paths']['patch_png_dir']
    uuid_file = config['paths']['uuid_name_file']
    print("[DEBUG] slide_folder:", slide_folder)
    print("[DEBUG] patch_folder:", patch_folder)
    print("[DEBUG] save_folder:", save_folder)
    print("[DEBUG] uuid_file:", uuid_file) 
    
    
    # Processing params
    define_patch_size = config['processing']['patch_size']
    patch_level = config['processing']['patch_level']

    # Scaling
    if define_patch_size == 256:
        scale = 1
    else:
        raise ValueError(f"Unsupported patch_size: {define_patch_size}")

    # Create save folder
    os.makedirs(save_folder, exist_ok=True)

    # Load UUID mapping
    uuid_data = pd.read_excel(uuid_file, engine='openpyxl', header=None)
    svs2uuid = {row[1].rstrip('\n'): row[0] for row in uuid_data.values}
    print(f"[INFO] Loaded UUID mapping with {len(svs2uuid)} entries.")

    # Get all H5 files
    all_file_names = [f for f in os.listdir(patch_folder) if f.endswith('.h5')]
    print(f"[INFO] Found {len(all_file_names)} H5 files to process.")

    if not all_file_names:
        print("[WARNING] No .h5 files found in patch folder.")
        return

    # Threaded processing
    pool = ThreadPoolExecutor(max_workers=4)
    for patch_file_name in all_file_names:
        future = pool.submit(
            generate_patch,
            patch_file_name,
            slide_folder,
            patch_folder,
            save_folder,
            svs2uuid,
            define_patch_size,
            scale
        )
        future.add_done_callback(handle_future)

    pool.shutdown(wait=True)
    print("[INFO] All patches processed.")

if __name__ == "__main__":
    main()
