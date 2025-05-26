import os
import argparse
import yaml
from tqdm import tqdm
import h5py
import openslide
from PIL import Image

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

    print(f"[INFO] Reading input patch of {input_size}x{input_size} → resized to {output_size}x{output_size}")
    return input_size, output_size


def save_patch_pngs(slide_path, coords, save_dir, patch_size, level, magnification):
    slide = openslide.OpenSlide(slide_path)
    os.makedirs(save_dir, exist_ok=True)
    resized_patch_size = get_patch_params(patch_size, magnification)

    count = 0
    for coord in tqdm(coords, desc=f"Saving patches for {os.path.basename(slide_path)}", leave=False, position=1):
        x, y = map(int, coord)
        patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
        patch = patch.resize((resized_patch_size, resized_patch_size))
        patch_name = f"{x}_{y}.png"
        patch.save(os.path.join(save_dir, patch_name))
        count += 1

    return count, resized_patch_size

def main(args, config):
    slide_dir = config['paths']['slide_dir']
    patch_h5_dir = config['paths']['patch_save_dir']
    patch_png_map = config['paths'].get('patch_png_dir', {})
    print(patch_png_map)
    key = f"patch_{args.patch_size}x{args.patch_size}_{args.magnification}"
    patch_png_dir = patch_png_map.get(key, None)
    if patch_png_dir is None:
        raise ValueError(f"patch_png_dir is missing entry for key: '{key}'")

    os.makedirs(patch_png_dir, exist_ok=True)

    h5_files = sorted([f for f in os.listdir(patch_h5_dir) if f.endswith('.h5')])
    for h5_file in tqdm(h5_files, desc="Processing slides", position=0):
        slide_id = h5_file.replace('.h5', '')
        h5_path = os.path.join(patch_h5_dir, h5_file)
        slide_path = os.path.join(slide_dir, slide_id + ".tif")

        if not os.path.exists(slide_path):
            print(f"---> Slide not found: {slide_path}")
            continue

        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]

        slide_patch_dir = os.path.join(patch_png_dir, slide_id)
        patch_count, patch_size_out = save_patch_pngs(
            slide_path, coords, slide_patch_dir,
            patch_size=args.patch_size, level=0,
            magnification=args.magnification
        )

        print(f"✅ Completed {slide_id}: {patch_count} patches saved at {patch_size_out}x{patch_size_out}px each in '{slide_patch_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--magnification', type=str, required=True)

    args = parser.parse_args()
    config = load_config(args.config)
    main(args, config)
