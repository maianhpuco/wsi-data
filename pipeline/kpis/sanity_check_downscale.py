import os
import glob
import argparse
import yaml
import cv2
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

def main(args):
    config = load_yaml_config(args.config)

    for split in ["train", "test", "validation"]:
        pattern = config["patterns"]["slide_dir"][split]
        save_root_base = config["sanity_downscale_dir"][split]
        slide_root = config["paths"]["slide_dir"][split]

        all_paths = glob.glob(pattern)
        print(f"[{split.upper()}] Found {len(all_paths)} TIFF files")

        for img_path in tqdm(all_paths):
            try:
                img = tiff.imread(img_path)
                class_name = Path(img_path).parts[-2]

                downscale_factor = get_downscale_factor(img.shape)
                img_downscaled = downscale_image(img, downscale_factor)

                relative_path = Path(img_path).relative_to(slide_root)
                save_path = Path(save_root_base).parent / Path(split) / Path("Task2_WSI_level") / class_name / relative_path.name
                save_path.parent.mkdir(parents=True, exist_ok=True)

                tiff.imwrite(str(save_path), img_downscaled)
                
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downscale pathology TIFFs for sanity check")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file") 
    args = parser.parse_args() 
    main(args)
