import json
import sqlite3
import hashlib
import os
import yaml
from shapely.geometry import Polygon
from pathlib import Path

try:
    from jnius import autoclass
except ImportError:
    print("  → [WARNING] pyjnius not installed. Run: pip install pyjnius")
    autoclass = None

def load_yaml_config(config_path):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"  → [ERROR] Failed to load config {config_path}: {e}")
        return None

def deserialize_java_object(geom_data):
    if autoclass is None:
        return None, "pyjnius not available"
    try:
        ByteArrayInputStream = autoclass('java.io.ByteArrayInputStream')
        ObjectInputStream = autoclass('java.io.ObjectInputStream')
        byte_stream = ByteArrayInputStream(geom_data)
        object_stream = ObjectInputStream(byte_stream)
        java_obj = object_stream.readObject()
        if java_obj.__class__.__name__ == 'Polygon':
            coords = [(java_obj.xpoints[i], java_obj.ypoints[i]) for i in range(java_obj.npoints)]
            return Polygon(coords), ""
        return None, f"Unsupported Java object: {java_obj.__class__.__name__}"
    except Exception as e:
        return None, f"Java deserialization failed: {e}"

def extract_annotations_as_json(db_path, image_dir, output_json_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT rdf.FILENAME, ra.DATA
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

    ann_dict = {}
    debug_count = 0
    for filename, geom_data in annotations:
        fname_key = Path(filename).stem
        try:
            geom, error = deserialize_java_object(geom_data)
            if geom is None:
                raise Exception(error)
            coords = list(map(list, geom.exterior.coords))
            ann_dict.setdefault(fname_key, []).append(coords)
        except Exception as e:
            if debug_count < 3:
                print(f"  → [DEBUG] Error parsing {filename}: {e}")
                debug_count += 1

    with open(output_json_path, 'w') as f:
        json.dump(ann_dict, f, indent=2)
        print(f"✅ Saved annotations to: {output_json_path}")

def main(config_path):
    config = load_yaml_config(config_path)
    if config is None:
        return
    db_path = config["paths"]["orbit_db"]
    output_json_path = config["paths"]["json_save_path"]
    image_dir = config["paths"]["slide_dir"]["train"]
    extract_annotations_as_json(db_path, image_dir, output_json_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract glomeruli annotations and save as JSON")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
