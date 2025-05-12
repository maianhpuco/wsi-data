"""
Script to retrieve patches for TCGA-Renal 4-shot learning dataset.
Reads a split CSV (tcga-renal_split_01.csv) and loads patches from .h5 files.
Returns patches as NumPy arrays for few-shot learning.
"""

import os
import pandas as pd
import h5py
import numpy as np
import logging
import argparse
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_patches(split_csv_path):
    """Load patches from .h5 files based on split CSV."""
    logger.info("Reading split CSV from %s", split_csv_path)
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")
    
    # Read split CSV
    df = pd.read_csv(split_csv_path)
    
    # Validate required columns
    required_columns = ['class_label', 'wsi_id', 'patch_id', 'patch_h5_path', 'patch_coord_x', 'patch_coord_y']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in split CSV: {col}")
    
    # Group by class and WSI
    data = {}
    for class_label in df['class_label'].unique():
        class_df = df[df['class_label'] == class_label]
        data[class_label] = {}
        
        for wsi_id in class_df['wsi_id'].unique():
            wsi_df = class_df[class_df['wsi_id'] == wsi_id]
            patches = []
            
            for _, row in wsi_df.iterrows():
                h5_path = row['patch_h5_path']
                patch_id = row['patch_id']
                
                try:
                    with h5py.File(h5_path, 'r') as h5_file:
                        patch_img = h5_file['imgs'][patch_id]
                        # Ensure patch is in RGB format (H, W, C)
                        if patch_img.shape[-1] != 3:
                            patch_img = np.transpose(patch_img, (1, 2, 0))
                        patches.append(patch_img)
                except Exception as e:
                    logger.error("Error loading patch %d from %s: %s", patch_id, h5_path, str(e))
                    continue
            
            if patches:
                data[class_label][wsi_id] = np.array(patches)
                logger.info("Loaded %d patches for %s/%s", len(patches), class_label, wsi_id)
            else:
                logger.warning("No patches loaded for %s/%s", class_label, wsi_id)
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Retrieve patches for TCGA-Renal 4-shot learning")
    parser.add_argument('--split_csv', type=str, required=True, 
                        help='Path to split CSV (e.g., tcga-renal_split_01.csv)')
    args = parser.parse_args()
    
    # Load patches
    data = load_patches(args.split_csv)
    
    # Print summary
    for class_label, wsis in data.items():
        logger.info("Class %s: %d WSIs, %d total patches", 
                    class_label, len(wsis), sum(len(patches) for patches in wsis.values()))

if __name__ == "__main__":
    main()