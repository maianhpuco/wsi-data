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

def compute_md5(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_annotations(db_path, image_dir, annotation_dir):
    os.makedirs(annotation_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image_md5, geometry FROM annotation")
    annotations = cursor.fetchall()

    # Group annotations by image hash
    ann_dict = {}
    for image_md5, geom_wkt in annotations:
        if image_md5 not in ann_dict:
            ann_dict[image_md5] = []
        ann_dict[image_md5].append(wkt.loads(geom_wkt))
    conn.close()

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".tiff")]
    total = len(image_files)

    for idx, image_file in enumerate(sorted(image_files), 1):
        print(f"[PROCESS] File {idx}/{total}: {image_file}")

        image_path = os.path.join(image_dir, image_file)
        md5 = compute_md5(image_path)

        if md5 not in ann_dict:
            print(f"  → [SKIP] No annotations for {image_file}")
            continue

        img = Image.open(image_path)
        width, height = img.size
        mask = np.zeros((height, width), dtype=np.uint8)

        for poly in ann_dict[md5]:
            if isinstance(poly, Polygon):
                pts = np.array(poly.exterior.coords).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)

        save_name = Path(image_file).stem + "_mask.png"
        mask_path = os.path.join(annotation_dir, save_name)
        Image.fromarray(mask).save(mask_path)
        print(f"  → Saved mask: {mask_path}")

def main(config_path):
    config = load_yaml_config(config_path)
    db_path = config["paths"]["orbit_db"]

    for split in ["train", "test"]:
        image_dir = config["paths"]["slide_dir"][split]
        annotation_dir = config["paths"]["anno_dir"][split]
        extract_annotations(db_path, image_dir, annotation_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract glomeruli annotations and save masks as PNG")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
