import sqlite3
import hashlib
import os
import yaml
from shapely import wkt
from shapely.geometry import Polygon
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def compute_md5(file_path):
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except (FileNotFoundError, IOError) as e:
        print(f"  → [ERROR] Failed to compute MD5 for {file_path}: {e}")
        return None

def load_yaml_config(config_path):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"  → [ERROR] Failed to load config {config_path}: {e}")
        return None

def validate_polygon(poly, width, height):
    if not isinstance(poly, Polygon) or not poly.is_valid:
        return False, "Invalid or non-Polygon geometry"
    bounds = poly.bounds
    if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > width or bounds[3] > height:
        return False, "Polygon out of image bounds"
    return True, ""

def estimate_memory(width, height):
    return (width * height * 1) / (1024 * 1024)

def preview_annotations(image, polygons, title="Preview"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    for poly in polygons:
        if isinstance(poly, Polygon):
            x, y = poly.exterior.xy
            plt.plot(x, y, color='red', linewidth=1)
    plt.title(title)
    plt.axis("off")
    plt.show()

def extract_annotations(db_path, image_dir, annotation_dir, preview=False):
    os.makedirs(annotation_dir, exist_ok=True)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print(cursor.fetchall())
        cursor.execute("SELECT image_md5, geometry FROM annotation")
        annotations = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"  → [ERROR] Database error: {e}")
        return
    finally:
        conn.close()

    ann_dict = {}
    for image_md5, geom_wkt in annotations:
        try:
            geom = wkt.loads(geom_wkt)
            ann_dict.setdefault(image_md5, []).append(geom)
        except Exception as e:
            print(f"  → [WARNING] Invalid WKT for {image_md5}: {e}")

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".tiff")]
    total = len(image_files)

    for idx, image_file in enumerate(sorted(image_files), 1):
        print(f"[PROCESS] File {idx}/{total}: {image_file}")

        image_path = os.path.join(image_dir, image_file)
        md5 = compute_md5(image_path)
        if md5 is None or md5 not in ann_dict:
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
        for poly in ann_dict[md5]:
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
    parser = argparse.ArgumentParser(description="Extract and preview glomeruli annotations as full-res masks")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--preview", action="store_true", help="Show annotation overlays before saving")
    args = parser.parse_args()
    main(args.config, preview=args.preview)
