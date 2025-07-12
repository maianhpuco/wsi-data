import os
import argparse
import yaml
from tqdm import tqdm
import h5py
import openslide
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_patch_params(magnification):
    output_size = 256
    if magnification == '5x':
        input_size = 2048  # Save all patches as 256x256 regardless of mag 
    elif magnification == '10x':
        input_size = 1024 # Read 1024x1024 to simulate410x 
    elif magnification == '20x':
        input_size = 512 # Read 512x512 to simulate 20x
    elif magnification == '40x':
        input_size = 256 # Read 512x512 to simulate 40x  
    else:
        raise ValueError(f"Unsupported magnification: {magnification}")
    return input_size, output_size

def build_slide_path_lookup(slide_name_file, uuid_name_file, slide_dir, ext=".tif"):
    slide_paths = {}
    
    df_names = pd.read_excel(slide_name_file)
    df_uuids = pd.read_excel(uuid_name_file)
    
    assert 'Filename' in df_names.columns and 'Filename' in df_uuids.columns and 'UUID' in df_uuids.columns

    uuid_map = dict(zip(df_uuids['Filename'], df_uuids['UUID']))
    
    for slide_name in df_names['Filename']:
        uuid = uuid_map.get(slide_name)
        if uuid:
            slide_path = os.path.join(slide_dir, uuid, slide_name)
            slide_id = slide_name.replace(ext, '')
            slide_paths[slide_id] = slide_path

    return slide_paths

def generate_patch(
    h5_file_name, slide_paths, patch_h5_dir, patch_png_dir, magnification):
    slide_id = h5_file_name.replace('.h5', '')
    slide_path = slide_paths.get(slide_id)
    h5_path = os.path.join(patch_h5_dir, h5_file_name)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    error_log_path = os.path.join(log_dir, f"error_path_gen_{magnification}.txt")

    if not slide_path or not os.path.exists(slide_path):
        print(f"[SKIP] Slide not found for: {slide_id}")
        with open(error_log_path, 'a') as f:
            f.write(f"{slide_id} - slide not found\n")
        return

    try:
        # Load coords first to know how many patches expected
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]
            patch_level = f['coords'].attrs.get('patch_level', 0)

        slide_patch_dir = os.path.join(patch_png_dir, slide_id)

        # Skip if all patches already exist
        if os.path.exists(slide_patch_dir):
            existing_pngs = [f for f in os.listdir(slide_patch_dir) if f.endswith('.png')]
            if len(existing_pngs) >= len(coords):  # Already processed
                print(f"[SKIP] {slide_id}: Already processed ({len(existing_pngs)} patches exist)")
                return

        # Continue to process
        input_size, output_size = get_patch_params(magnification)
        slide = openslide.OpenSlide(slide_path)
        os.makedirs(slide_patch_dir, exist_ok=True)

        for coord in tqdm(coords, desc=f"==> {slide_id}", leave=False):
            x, y = map(int, coord)
            out_path = os.path.join(slide_patch_dir, f"{x}_{y}.png")
            if os.path.exists(out_path):
                continue  # Skip patch if already saved

            patch = slide.read_region((x, y), patch_level, (input_size, input_size)).convert('RGB')
            patch = patch.resize((output_size, output_size))
            patch.save(out_path)

        print(f"[DONE] {slide_id}: {len(coords)} patches saved in {slide_patch_dir}")

    except Exception as e:
        print(f"[ERROR] {slide_id}: {e}")
        with open(error_log_path, 'a') as f:
            f.write(f"{slide_id} - {str(e)}\n")
            
def main(args, config):
    patch_h5_dir = config['paths']['patch_h5_dir']
    patch_png_map = config['paths'].get('patch_png_dir', {})
    slide_dir = config['paths']['source_dir']
    slide_name_file = config['paths']['slide_name_file']
    uuid_name_file = config['paths']['uuid_name_file']
    slide_ext = config.get('feature_extraction', {}).get('slide_ext', '.svs')

    # Load allowed slide_ids from CSV
    slide_filter_csv = config['paths'].get('slide_filter_csv', None)
    allowed_slide_ids = None
    if slide_filter_csv and os.path.exists(slide_filter_csv):
        df_filter = pd.read_csv(slide_filter_csv)
        allowed_slide_ids = set(df_filter['slide_id'].astype(str).str.replace(slide_ext, '', regex=False))

    key = f"patch_{args.patch_size}x{args.patch_size}_{args.magnification}"
    patch_png_dir = patch_png_map.get(key)
    if patch_png_dir is None:
        raise ValueError(f"Missing patch_png_dir for key: '{key}'")
    os.makedirs(patch_png_dir, exist_ok=True)

    slide_paths = build_slide_path_lookup(slide_name_file, uuid_name_file, slide_dir, ext=slide_ext)

    h5_files = sorted([f for f in os.listdir(patch_h5_dir) if f.endswith('.h5')])
    if allowed_slide_ids is not None:
        h5_files = [f for f in h5_files if f.replace('.h5', '') in allowed_slide_ids]

    with ThreadPoolExecutor(max_workers=16) as executor:
        for h5_file in tqdm(h5_files, desc="Dispatching patch generation"):
            executor.submit(generate_patch, h5_file, slide_paths, patch_h5_dir, patch_png_dir, args.magnification)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--magnification', type=str, required=True)

    args = parser.parse_args()
    config = load_config(args.config)
    main(args, config)
