import os
import glob
import argparse
import yaml
import cv2
import shutil
from tqdm import tqdm
import tifffile as tiff
from pathlib import Path

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_downscale_factor(image_shape, target_max_dim=512):
    h, w = image_shape[:2]
    return max(h, w) // target_max_dim if max(h, w) > target_max_dim else 1

def downscale_image(image, factor):
    if factor == 1:
        return image
    return cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor), interpolation=cv2.INTER_AREA)

def clear_and_create_dir(path):
    if os.path.exists(path):
        print(f"Clearing existing directory: {path}")
        shutil.rmtree(path)
    os.makedirs(path)
    print(f"Created directory: {path}")

def save_as_png(img, save_path):
    if len(img.shape) == 2:
        return cv2.imwrite(str(save_path), img)
    else:
        return cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def process_images(config, split):
    slide_root = config["paths"]["slide_dir"][split]
    save_root_base = config["paths"]["sanity_downscale_dir"][split]

    all_slide_paths = sorted(glob.glob(os.path.join(slide_root, "*.tiff")))
    print(f"[{split.upper()}] Found {len(all_slide_paths)} TIFF files")

    clear_and_create_dir(save_root_base)

    for slide_path in tqdm(all_slide_paths):
        try:
            filename = Path(slide_path).stem
            save_path = Path(save_root_base) / f"{filename}.png"

            img = tiff.imread(slide_path)
            factor = get_downscale_factor(img.shape)
            img_downscaled = downscale_image(img, factor)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_as_png(img_downscaled, save_path):
                print(f"Saved downscaled PNG: {save_path}")
            else:
                print(f"[ERROR] Failed to save PNG: {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed to process {slide_path}: {type(e).__name__} - {e}")

def main(args):
    config = load_yaml_config(args.config)
    for split in ["train", "test"]:
        process_images(config, split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downscale Glomeruli TIFFs and save as PNGs")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
