import sqlite3
import hashlib
import os
import yaml
from shapely import wkt, wkb
from shapely.geometry import Polygon
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def compute_md5(file_path):
    """Compute MD5 hash of a file."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except (FileNotFoundError, IOError) as e:
        print(f"  → [ERROR] Failed to compute MD5 for {file_path}: {e}")
        return None

def load_yaml_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"  → [ERROR] Failed to load config {config_path}: {e}")
        return None

def validate_polygon(poly, width, height):
    """Validate a polygon: check if it's a valid Polygon and within image bounds."""
    if not isinstance(poly, Polygon) or not poly.is_valid:
        return False, "Invalid or non-Polygon geometry"
    bounds = poly.bounds
    if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > width or bounds[3] > height:
        return False, "Polygon out of image bounds"
    return True, ""

def estimate_memory(width, height):
    """Estimate memory required for a mask in MB."""
    return (width * height * 1) / (1024 * 1024)

def preview_annotations(image, polygons, title="Preview", max_size=1000):
    """Preview annotations overlaid on a downscaled image."""
    img_array = np.array(image)
    
    # Downscale image for preview if too large
    height, width = img_array.shape[:2]
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width, new_height = int(width * scale), int(height * scale)
        img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
        scaled_polygons = []
        for poly in polygons:
            scaled_coords = [(x * scale, y * scale) for x, y in poly.exterior.coords]
            scaled_polygons.append(Polygon(scaled_coords))
    else:
        scaled_polygons = polygons

    plt.figure(figsize=(8, 8))
    plt.imshow(img_array)
    for poly in scaled_polygons:
        if isinstance(poly, Polygon):
            x, y = poly.exterior.xy
            plt.plot(x, y, color='red', linewidth=1)
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{title.replace(' ', '_')}_preview.png")
    plt.close()

def extract_annotations(db_path, image_dir, annotation_dir, preview=False):
    """Extract annotations from database and save as binary PNG masks."""
    os.makedirs(annotation_dir, exist_ok=True)

    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Print distinct RAW_ANNOTATION_TYPE values
        cursor.execute("SELECT DISTINCT RAW_ANNOTATION_TYPE FROM RAW_ANNOTATION")
        types = cursor.fetchall()
        print(f"  → [DEBUG] Available RAW_ANNOTATION_TYPE values: {types}")

        # Query using FILENAME for matching
        cursor.execute("""
            SELECT rdf.FILENAME, ra.DATA, ra.RAW_ANNOTATION_TYPE 
            FROM RAW_ANNOTATION ra
            JOIN RAW_DATA_FILE rdf ON ra.RAW_DATA_FILE_ID = rdf.RAW_DATA_FILE_ID
            WHERE ra.DATA IS NOT NULL
        """)
        annotations = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"  → [ERROR] Database error: {e}")
        return
    finally:
        conn.close()

    # Group annotations by filename (without extension)
    ann_dict = {}
    debug_printed = False
    for filename, geom_data, ann_type in annotations:
        try:
            # Try parsing DATA as WKB (or skip parsing until format is confirmed)
            geom = wkb.loads(geom_data)
            # Use filename without extension for matching
            fname_key = Path(filename).stem
            ann_dict.setdefault(fname_key, []).append(geom)
        except Exception as e:
            if not debug_printed:
                # Print sample DATA as hex for debugging
                print(f"  → [DEBUG] Sample DATA (hex) for {filename}: {geom_data.hex()[:100]}...")
                debug_printed = True
            print(f"  → [WARNING] Invalid geometry for {filename} (type {ann_type}): {e}")

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".tiff")]
    total = len(image_files)

    for idx, image_file in enumerate(sorted(image_files), 1):
        print(f"[PROCESS] File {idx}/{total}: {image_file}")

        image_path = os.path.join(image_dir, image_file)
        # Use filename without extension for matching
        fname_key = Path(image_file).stem
        if fname_key not in ann_dict:
            print(f"  → [SKIP] No annotations for {image_file}")
            continue

        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            print(f"  → [ERROR] Failed to open image {image_file}: {e}")
            continue

        mem_mb = estimate_memory(width, height)
        if mem_mb > 1000:
            print(f"  → [WARNING] Large image ({width}x{height}, ~{mem_mb:.2f} MB).")

        try:
            mask = np.zeros((height, width), dtype=np.uint8)
        except MemoryError as e:
            print(f"  → [ERROR] Memory error: {e}")
            continue

        valid_polygons = []
        for poly in ann_dict[fname_key]:
            is_valid, error_msg = validate_polygon(poly, width, height)
            if not is_valid:
                print(f"  → [WARNING] Skipping polygon: {error_msg}")
                continue
            try:
                pts = np.array(poly.exterior.coords).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
                valid_polygons.append(poly)
            except Exception as e:
                print(f"  → [WARNING] Polygon fill failed: {e}")

        if not valid_polygons:
            print(f"  → [SKIP] No valid polygons for {image_file}")
            continue

        if preview:
            preview_annotations(img, valid_polygons, title=image_file)

        save_name = Path(image_file).stem + "_mask.png"
        mask_path = os.path.join(annotation_dir, save_name)
        try:
            Image.fromarray(mask).save(mask_path, optimize=True)
            print(f"  → Saved mask: {mask_path}")
        except (MemoryError, IOError) as e:
            print(f"  → [ERROR] Failed to save mask: {e}")

def main(config_path, preview=False):
    config = load_yaml_config(config_path)
    if config is None:
        return

    db_path = config["paths"]["orbit_db"]
    for split in ["train", "test"]:
        image_dir = config["paths"]["slide_dir"][split]
        annotation_dir = config["paths"]["anno_dir"][split]
        print(f"\n[INFO] Processing {split} split...")
        extract_annotations(db_path, image_dir, annotation_dir, preview=preview)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and preview glomeruli annotations as full-res PNG masks")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--preview", action="store_true", help="Save annotation overlay previews")
    args = parser.parse_args()
    main(args.config, preview=args.preview)