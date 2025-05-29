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
        input_size = 2048
    elif magnification == '10x':
        input_size = 1024
    elif magnification == '20x':
        input_size = 512
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

def generate_patch(h5_file_name, slide_paths, patch_h5_dir, patch_png_dir, magnification):
    slide_id = h5_file_name.replace('.h5', '')
    slide_path = slide_paths.get(slide_id)
    h5_path = os.path.join(patch_h5_dir, h5_file_name)
    # print("finding", slide_path)
    if not slide_path or not os.path.exists(slide_path):
        print(f"[SKIP] Slide not found for: {slide_id}")
        return

    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        patch_level = f['coords'].attrs.get('patch_level', 0)

    input_size, output_size = get_patch_params(magnification)
    slide = openslide.OpenSlide(slide_path)
    slide_patch_dir = os.path.join(patch_png_dir, slide_id)
    os.makedirs(slide_patch_dir, exist_ok=True)

    for coord in tqdm(coords, desc=f"==> {slide_id}", leave=False):
        x, y = map(int, coord)
        patch = slide.read_region((x, y), patch_level, (input_size, input_size)).convert('RGB')
        patch = patch.resize((output_size, output_size))
        patch.save(os.path.join(slide_patch_dir, f"{x}_{y}.png"))

    print(f"[DONE] {slide_id}: {len(coords)} patches saved in {slide_patch_dir}")

def main(args, config):
    patch_h5_dir = config['paths']['patch_h5_dir']
    patch_png_map = config['paths'].get('patch_png_dir', {})
    slide_dir = config['paths']['source_dir']
    slide_name_file = config['paths']['slide_name_file']
    uuid_name_file = config['paths']['uuid_name_file']
    slide_ext = config.get('feature_extraction', {}).get('slide_ext', '.svs')

    key = f"patch_{args.patch_size}x{args.patch_size}_{args.magnification}"
    patch_png_dir = patch_png_map.get(key)
    if patch_png_dir is None:
        raise ValueError(f"Missing patch_png_dir for key: '{key}'")
    os.makedirs(patch_png_dir, exist_ok=True)

    # Build slide ID → path map
    slide_paths = build_slide_path_lookup(slide_name_file, uuid_name_file, slide_dir, ext=slide_ext)

    # Process each .h5 patch file
    h5_files = sorted([f for f in os.listdir(patch_h5_dir) if f.endswith('.h5')])
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
