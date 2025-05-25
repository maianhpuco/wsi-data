import os
import sys
import glob
import pandas as pd
import numpy as np
import argparse
import h5py
import yaml
from tqdm import tqdm
from shapely.geometry import Polygon, Point
from rtree import index
import xml.etree.ElementTree as ET

PATCH_SIZE = 256

def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def extract_coordinates(file_path):
    root = parse_xml(file_path)
    if root is None:
        return None

    contours = []
    for annotation in root.findall(".//Annotation"):
        contour = []
        for coordinate in annotation.findall(".//Coordinate"):
            x = coordinate.attrib.get("X")
            y = coordinate.attrib.get("Y")
            if x and y:
                contour.append((float(x), float(y)))

        if contour and contour[0] != contour[-1]:
            contour.append(contour[0])
        if contour:
            contours.append(contour)

    if not contours:
        return None

    polygons = [Polygon(contour) for contour in contours if len(contour) > 2]
    return polygons

def check_xy_in_coordinates_from_topleft(polygons, coordinates_h5):
    label = np.zeros(len(coordinates_h5), dtype=np.int8)
    spatial_index = index.Index((i, poly.bounds, None) for i, poly in enumerate(polygons))

    for i, (x, y) in enumerate(coordinates_h5):
        center_x = x + PATCH_SIZE / 2
        center_y = y + PATCH_SIZE / 2
        patch_center = Point(center_x, center_y)

        candidates = spatial_index.intersection((center_x, center_y, center_x, center_y))
        if any(polygons[j].contains(patch_center) for j in candidates):
            label[i] = 1

    return label

def read_h5_data(file_path):
    with h5py.File(file_path, "r") as f:
        return f["coords"][:]

def main(args):
    xml_files = sorted([f for f in os.listdir(args.annotation_path) if f.endswith(".xml")])
    h5_files = set(f for f in os.listdir(args.features_h5_path) if f.endswith(".h5"))

    to_process = [f for f in xml_files if f.replace(".xml", ".h5") in h5_files]
    print("Total files to process:", len(to_process))

    for idx, xml_filename in enumerate(to_process):
        name = os.path.splitext(xml_filename)[0]
        print(f"[{idx+1}/{len(to_process)}] Processing {name}")

        h5_path = os.path.join(args.features_h5_path, f"{name}.h5")
        xml_path = os.path.join(args.annotation_path, xml_filename)

        polygons = extract_coordinates(xml_path)
        if polygons is None:
            print(f"No valid annotation found for {name}, skipping.")
            continue

        coordinates = read_h5_data(h5_path)
        mask = check_xy_in_coordinates_from_topleft(polygons, coordinates)

        mask_path = os.path.join(args.ground_truth_path, f"{name}.npy")
        print("Shape of mask:", mask.shape)
        np.save(mask_path, mask)
        print(f"Saved mask to: {mask_path}")

    final_masks = glob.glob(os.path.join(args.ground_truth_path, "*.npy"))
    print("Generated mask files:", len(final_masks))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.features_h5_path = config['paths']['ht_files']
    args.annotation_path = config['paths']['anno_xml_dir']
    args.ground_truth_path = config['paths']['gt_h5_dir']

    os.makedirs(args.ground_truth_path, exist_ok=True)
    main(args)
