import os
import argparse
import yaml
from tqdm import tqdm
import h5py
import openslide
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_patch_params(magnification):
    """
    Return the patch read size (input size) based on magnification.
    Output patch is always resized to 256x256.
    """
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

def generate_patch(h5_file_name, slide_dir, patch_h5_dir, patch_png_dir, magnification):
    slide_id = h5_file_name.replace('.h5', '')
    slide_path = os.path.join(slide_dir, slide_id + ".tif")
    h5_path = os.path.join(patch_h5_dir, h5_file_name)

    if not os.path.exists(slide_path):
        print(f" Slide not found: {slide_path}")
        return

    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        patch_level = f['coords'].attrs.get('patch_level', 0)

    input_size, output_size = get_patch_params(magnification)
    slide = openslide.OpenSlide(slide_path)
    slide_patch_dir = os.path.join(patch_png_dir, slide_id)
    os.makedirs(slide_patch_dir, exist_ok=True)

    for coord in tqdm(coords, desc=f"🔄 {slide_id}", leave=False):
        x, y = map(int, coord)
        patch = slide.read_region((x, y), patch_level, (input_size, input_size)).convert('RGB')
        patch = patch.resize((output_size, output_size))
        patch_name = f"{x}_{y}.png"
        patch.save(os.path.join(slide_patch_dir, patch_name))

    print(f"==>  Completed {slide_id}: {len(coords)} patches saved at {output_size}x{output_size}px in '{slide_patch_dir}'")

def main(args, config):
    slide_dir = config['paths']['slide_dir']
    patch_h5_dir = config['paths']['patch_save_dir']
    patch_png_map = config['paths'].get('patch_png_dir', {})

    key = f"patch_{args.patch_size}x{args.patch_size}_{args.magnification}"
    patch_png_dir = patch_png_map.get(key, None)
    if patch_png_dir is None:
        raise ValueError(f"patch_png_dir is missing entry for key: '{key}'")

    os.makedirs(patch_png_dir, exist_ok=True)

    h5_files = sorted([f for f in os.listdir(patch_h5_dir) if f.endswith('.h5')])
    with ThreadPoolExecutor(max_workers=16) as executor:
        for h5_file in tqdm(h5_files, desc="Dispatching patch generation"):
            executor.submit(generate_patch, h5_file, slide_dir, patch_h5_dir, patch_png_dir, args.magnification)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--patch_size', type=int, required=True)  # must be 256
    parser.add_argument('--magnification', type=str, required=True)  # 5x, 10x, or 20x

    args = parser.parse_args()
    config = load_config(args.config)
    main(args, config)
