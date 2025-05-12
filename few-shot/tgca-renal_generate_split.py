"""
Script to generate a 4-shot learning split for TCGA-Renal (KIRC, KICH, KIRP).
Selects 4 WSIs per class and 16 patches per WSI from pre-generated .h5 files.
Saves split information in a CSV file (tcga-renal_split_01.csv).
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import h5py
import random
import logging
import argparse
from pathlib import Path

# Set project root for importing custom modules
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from a YAML file."""
    logger.info("Loading configuration from %s", config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Validate required fields
    required_paths = ['source_dir', 'patch_h5_dir', 'uuid_name_file']
    required_processing = ['patch_size', 'patch_level', 'num_patches', 'n_shot']
    
    for field in required_paths:
        if field not in config.get('paths', {}):
            raise ValueError(f"Missing required path field: {field}")
    
    for field in required_processing:
        if field not in config.get('processing', {}):
            raise ValueError(f"Missing required processing field: {field}")
    
    return config

def load_wsi_list(uuid_name_file):
    """Load WSI filenames and UUID mappings from uuid_name_file."""
    logger.info("Loading WSI list from %s", uuid_name_file)
    try:
        all_data = pd.read_excel(uuid_name_file, engine='openpyxl', header=None)
        slides = all_data[1].tolist()
        id_names = dict(zip(all_data[1], all_data[0]))
        return slides, id_names
    except Exception as e:
        raise ValueError(f"Failed to read uuid_name_file {uuid_name_file}: {str(e)}")

def select_4shot_wsis(slides, n_shot):
    """Select n_shot WSIs randomly."""
    logger.info("Selecting %d WSIs", n_shot)
    if len(slides) < n_shot:
        raise ValueError(f"Not enough WSIs: {len(slides)} available, {n_shot} required")
    
    return random.sample(slides, n_shot)

def select_patches(h5_path, num_patches):
    """Select num_patches random patches from an .h5 file."""
    try:
        logger.info("Selecting %d patches from %s", num_patches, h5_path)
        with h5py.File(h5_path, 'r') as h5_file:
            patch_imgs = h5_file['imgs'][:]
            patch_coords = h5_file['coords'][:]
        
        if len(patch_imgs) < num_patches:
            logger.warning("Not enough patches in %s: %d available, %d required", h5_path, len(patch_imgs), num_patches)
            return [], []
        
        indices = random.sample(range(len(patch_imgs)), num_patches)
        selected_coords = patch_coords[indices]
        return indices, selected_coords
    except Exception as e:
        logger.error("Error selecting patches from %s: %s", h5_path, str(e))
        return [], []

def generate_split(config_paths, output_dir):
    """Generate a 4-shot learning split and save to CSV."""
    split_data = []
    
    for config_path in config_paths:
        # Load configuration
        config = load_config(config_path)
        paths = config['paths']
        processing = config['processing']
        
        # Extract configuration parameters
        source_dir = paths['source_dir']
        patch_h5_dir = paths['patch_h5_dir']
        uuid_name_file = paths['uuid_name_file']
        class_label = os.path.basename(config_path).replace("data_", "").replace(".yaml", "")  # e.g., kirc
        num_patches = processing['num_patches']
        n_shot = processing['n_shot']
        
        # Load WSI list
        slides, id_names = load_wsi_list(uuid_name_file)
        
        # Select n_shot WSIs
        selected_slides = select_4shot_wsis(slides, n_shot)
        
        # Process each WSI
        for slide in selected_slides:
            slide_id, _ = os.path.splitext(slide)
            wsi_path = os.path.join(source_dir, id_names[slide], slide)
            h5_path = os.path.join(patch_h5_dir, f"{slide_id}.h5")
            
            if not os.path.exists(wsi_path):
                logger.warning("WSI not found: %s", wsi_path)
                continue
            
            if not os.path.exists(h5_path):
                logger.warning("H5 file not found: %s", h5_path)
                continue
            
            # Select patches
            patch_indices, patch_coords = select_patches(h5_path, num_patches)
            if not patch_indices:
                logger.warning("No patches selected for %s", slide_id)
                continue
            
            # Record split information
            for idx, coord in zip(patch_indices, patch_coords):
                split_data.append({
                    'class_label': class_label.upper(),  # e.g., KIRC
                    'wsi_id': slide_id,
                    'wsi_path': wsi_path,
                    'patch_id': idx,
                    'patch_h5_path': h5_path,
                    'patch_coord_x': coord[0],
                    'patch_coord_y': coord[1]
                })
    
    # Save split CSV
    os.makedirs(output_dir, exist_ok=True)
    split_csv_path = os.path.join(output_dir, 'tcga-renal_split_01.csv')
    split_df = pd.DataFrame(split_data)
    split_df.to_csv(split_csv_path, index=False)
    logger.info("Split CSV saved to %s", split_csv_path)
    
    # Log summary
    summary = split_df.groupby('class_label').agg({
        'wsi_id': 'nunique',
        'patch_id': 'count'
    }).rename(columns={'wsi_id': 'num_wsis', 'patch_id': 'num_patches'})
    logger.info("Split summary:\n%s", summary)
    
##---------generate 1 slit (can loop this to generate multiple splits)---------------- 
def main():
    parser = argparse.ArgumentParser(description="Generate 4-shot learning split for TCGA-Renal")
    parser.add_argument('--configs', type=str, nargs='+', required=True, 
                        help='List of paths to YAML configuration files (e.g., data_kirc.yaml data_kich.yaml data_kirp.yaml)')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for split CSV')
    args = parser.parse_args()
    
    generate_split(args.configs, args.output_dir)

    
# def main():
#     parser = argparse.ArgumentParser(description="Generate multiple 4-shot learning splits for TCGA-Renal")
#     parser.add_argument('--configs', type=str, nargs='+', required=True, 
#                         help='List of paths to YAML configuration files (e.g., data_kirc.yaml data_kich.yaml data_kirp.yaml)')
#     parser.add_argument('--output_dir', type=str, required=True, 
#                         help='Output directory for split CSVs')
#     parser.add_argument('--num_splits', type=int, default=5,
#                         help='Number of few-shot splits to generate (default: 5)')
#     args = parser.parse_args()

#     for split_id in range(1, args.num_splits + 1):
#         logger.info(f"Generating split {split_id}/{args.num_splits}")
#         generate_split(args.configs, args.output_dir, split_id)

if __name__ == "__main__":
    main()