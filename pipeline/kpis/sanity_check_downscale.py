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

def main(args):
    config = load_yaml_config(args.config)

    for split in ["train", "test", "validation"]:
        pattern = config["patterns"]["slide_dir"][split]
        save_root_base = config["paths"]["sanity_downscale_dir"][split]
        slide_root = config["paths"]["slide_dir"][split]

        all_paths = glob.glob(pattern)
        print(f"[{split.upper()}] Found {len(all_paths)} TIFF files")

        # Clear target save directory once per split
        clear_and_create_dir(save_root_base)

        for img_path in tqdm(all_paths):
            try:
                img = tiff.imread(img_path)
                class_name = Path(img_path).parts[-2]

                downscale_factor = get_downscale_factor(img.shape)
                img_downscaled = downscale_image(img, downscale_factor)

                relative_path = Path(img_path).relative_to(slide_root)
                save_path = Path(save_root_base) / class_name / relative_path.with_suffix(".png").name
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # Save as PNG using OpenCV
                if len(img_downscaled.shape) == 2:  # grayscale
                    success = cv2.imwrite(str(save_path), img_downscaled)
                else:  # multi-channel, convert RGB if needed
                    success = cv2.imwrite(str(save_path), cv2.cvtColor(img_downscaled, cv2.COLOR_RGB2BGR))

                if success:
                    print(f"Saved downscaled PNG to: {save_path}")
                else:
                    print(f"Failed to save PNG at: {save_path}")

            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downscale pathology TIFFs and save as PNGs")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
