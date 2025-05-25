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
    if len(img.shape) == 2:  # grayscale
        success = cv2.imwrite(str(save_path), img)
    else:
        success = cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return success

def process_images(config, split):
    slide_pattern = config["patterns"]["slide_dir"][split]
    mask_pattern = config["patterns"]["anno_xml_dir"][split]
    slide_root = config["paths"]["slide_dir"][split]
    mask_root = config["paths"]["anno_xml_dir"][split]
    save_root_base = config["paths"]["sanity_downscale_dir"][split]

    all_slide_paths = glob.glob(slide_pattern)
    all_mask_paths = glob.glob(mask_pattern)
    print(f"[{split.upper()}] Found {len(all_slide_paths)} WSI files and {len(all_mask_paths)} Mask files")

    clear_and_create_dir(save_root_base)

    for slide_path in tqdm(all_slide_paths):
        try:
            class_name = Path(slide_path).parts[-2]
            relative_slide = Path(slide_path).relative_to(slide_root)
            slide_png_path = Path(save_root_base) / class_name / relative_slide.with_suffix(".png").name

            img = tiff.imread(slide_path)
            factor = get_downscale_factor(img.shape)
            img_downscaled = downscale_image(img, factor)
            slide_png_path.parent.mkdir(parents=True, exist_ok=True)

            if save_as_png(img_downscaled, slide_png_path):
                print(f"Saved slide PNG: {slide_png_path}")
            else:
                print(f"[ERROR] Failed saving slide: {slide_png_path}")

            # Corresponding mask
            mask_path = slide_path.replace("_wsi.tiff", "_mask.tiff")
            if os.path.exists(mask_path):
                mask = tiff.imread(mask_path)
                mask_downscaled = downscale_image(mask, factor)
                mask_png_path = slide_png_path.with_name(slide_png_path.name.replace("_wsi.png", "_mask.png"))
                if save_as_png(mask_downscaled, mask_png_path):
                    print(f"Saved mask PNG: {mask_png_path}")
                else:
                    print(f"[ERROR] Failed saving mask: {mask_png_path}")
            else:
                print(f"[WARNING] No mask found for: {slide_path}")

        except Exception as e:
            print(f"[ERROR] Failed to process {slide_path}: {type(e).__name__} - {e}")

def main(args):
    config = load_yaml_config(args.config)
    for split in ["train", "test", "validation"]:
        process_images(config, split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downscale pathology TIFFs and save as PNGs (with masks)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
