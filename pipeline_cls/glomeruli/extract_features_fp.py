import os
import sys
import argparse
import yaml
import time
from functools import partial
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
import numpy as np
import pandas as pd

# Ensure CLAM is in the import path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

# Check CUDA availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print(f"CUDA is available. Using: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Using CPU only.")

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_w_loader(output_path, loader, model, verbose=0):
    """Compute features for a DataLoader and save to HDF5."""
    if verbose > 0:
        print(f'Processing a total of {len(loader)} batches')

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():
            batch = data['img']
            coords = data['coord'].numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True)
            
            features = model(batch)
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
    
    return output_path

def process_split(split, cfg):
    """Process feature extraction for a given split (train/test)."""
    # Set paths from config
    source = cfg['paths']['slide_dir'][split]
    patch_h5_dir = cfg['paths']['patch_dir'][split]
    feat_dir = os.path.join(cfg['paths']['save_dir'], split, 'features_fp')
    csv_path = os.path.join(cfg['paths']['save_dir'], split, 'slide_list.csv')

    # Create necessary directories
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(os.path.join(feat_dir خام

System: I notice that the response was cut off. Let me complete the modified code for you, ensuring it adapts to the new paths specified in the configuration and processes both `train` and `test` splits. I'll include the full script with the necessary changes, wrapped in an `<xaiArtifact>` tag with a new UUID, as this is a continuation but still a distinct modification from the previous artifacts.

### Changes to Be Made:
1. **Update Path Handling**:
   - Replace the single `source` path with `paths['slide_dir'][split]` to process `train` and `test` splits separately.
   - Use `paths['patch_dir'][split]` for `patch_h5_dir`.
   - Adjust `feat_dir` and `csv_path` to be split-specific (e.g., `save_dir/train/features_fp` and `save_dir/train/slide_list.csv` for the train split).
2. **Iterate Over Splits**:
   - Modify the `main` function to loop over `train` and `test` splits, processing each independently.
3. **Ensure Config Compatibility**:
   - Ensure the script uses the new `slide_dir` and `patch_dir` paths from the config:
     - `slide_dir`:
       - Train: `/home/mvu9/processing_datasets/glomeruli_pyramidal/train`
       - Test: `/home/mvu9/processing_datasets/glomeruli_pyramidal/test`
     - `patch_dir`:
       - Train: `/home/mvu9/processing_datasets/glomeruli/patches/train`
       - Test: `/home/mvu9/processing_datasets/glomeruli/patches/test`
4. **Maintain Other Paths**:
   - The `save_dir` will be used as the base for `feat_dir` and `csv_path`, but split-specific subdirectories will be created.
   - The `raw_slide_dir`, `anno_dir`, `sanity_downscale_dir`, `orbit_db`, and `json_save_path` are not used in this script but should be present in the config for compatibility with other scripts.

### Modified Code:

<xaiArtifact artifact_id="abb31952-cfec-47f9-9404-f763fff3378f" artifact_version_id="bc4164c0-8b54-4cc0-b098-7fd936584904" title="feature_extraction.py" contentType="text/python">
import os
import sys
import argparse
import yaml
import time
from functools import partial
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
import numpy as np
import pandas as pd

# Ensure CLAM is in the import path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

# Check CUDA availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print(f"CUDA is available. Using: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Using CPU only.")

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_w_loader(output_path, loader, model, verbose=0):
    """Compute features for a DataLoader and save to HDF5."""
    if verbose > 0:
        print(f'Processing a total of {len(loader)} batches')

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():
            batch = data['img']
            coords = data['coord'].numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True)
            
            features = model(batch)
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
    
    return output_path

def process_split(split, cfg):
    """Process feature extraction for a given split (train/test)."""
    # Set paths from config
    source = cfg['paths']['slide_dir'][split]
    patch_h5_dir = cfg['paths']['patch_dir'][split]
    feat_dir = os.path.join(cfg['paths']['save_dir'], split, 'features_fp')
    csv_path = os.path.join(cfg['paths']['save_dir'], split, 'slide_list.csv')

    # Create necessary directories
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)

    # Generate slide list CSV
    print(f" Generating slide list CSV for {split} at: {csv_path}")
    slide_ext = cfg.get("feature_extraction", {}).get("slide_ext", ".tif")
    print(f"---- extension: {slide_ext}")
    slide_files = [f for f in os.listdir(source) if f.endswith(slide_ext)]
    print(f">>> Found: {len(slide_files)} slides in {split}")
    if not slide_files:
        print(f" No slides found in {source} with extension {slide_ext}")
        return
    print(f"Found {len(slide_files)} slides: {slide_files[:5]}")  # Print first 5 for debugging
    with open(csv_path, 'w') as f:
        f.write("slide_id\n")
        for s in slide_files:
            f.write(s + '\n')

    # Feature extraction config
    feat_cfg = cfg.get("feature_extraction", {})
    model_name = feat_cfg.get("model_name", "resnet50_trunc")
    batch_size = feat_cfg.get("batch_size", 256)
    target_patch_size = feat_cfg.get("target_patch_size", 224)
    slide_ext = feat_cfg.get("slide_ext", ".tif")
    no_auto_skip = feat_cfg.get("no_auto_skip", False)
    
    # Preprocessing config
    patch_size = cfg['processing']['patch_size']
    patch_level = cfg['processing']['patch_level']
    
    # Initialize dataset
    print(f'Initializing dataset for {split}')
    if csv_path is None:
        raise ValueError("CSV path is not provided")
    
    bags_dataset = Dataset_All_Bags(csv_path)
    total = len(bags_dataset)
    if total == 0:
        print(f" No slides found in dataset from {csv_path}")
        return
    print(f"Total slides in {split} dataset: {total}")

    # Load model
    model, img_transforms = get_encoder(model_name, target_img_size=target_patch_size)
    model.eval()
    model = model.to(device)
    
    # Check for existing feature files
    dest_files = os.listdir(os.path.join(feat_dir, 'pt_files'))
    
    # DataLoader kwargs
    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    # Process each slide
    for bag_candidate_idx in tqdm(range(total), desc=f"Processing {split}"):
        slide_id = bags_dataset[bag_candidate_idx].split(slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(patch_h5_dir, bag_name)
        slide_file_path = os.path.join(source, slide_id + slide_ext)
        print(f'\nProgress: {bag_candidate_idx+1}/{total}')
        print(f"Slide ID: {slide_id}")

        # Skip if features already exist and no_auto_skip is False
        if not no_auto_skip and slide_id + '.pt' in dest_files:
            print(f"Skipped {slide_id} (features already exist)")
            continue 
            
        # Verify H5 and slide file existence
        if not os.path.exists(h5_file_path):
            print(f" H5 file not found: {h5_file_path}")
            continue
        if not os.path.exists(slide_file_path):
            print(f" Slide file not found: {slide_file_path}")
            continue

        # Process slide
        output_path = os.path.join(feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        try:
            wsi = openslide.open_slide(slide_file_path)
            dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
                                        wsi=wsi, 
                                        img_transforms=img_transforms)
            
            if len(dataset) == 0:
                print(f" No patches found in {h5_file_path}")
                continue

            loader = DataLoader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
            output_file_path = compute_w_loader(output_path, loader=loader, model=model, verbose=1)

            time_elapsed = time.time() - time_start
            print(f'\nComputing features for {output_file_path} took {time_elapsed:.2f} s')

            # Save features as .pt
            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
                print(f'Features size: {features.shape}')
                print(f'Coordinates size: {file["coords"].shape}')

            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(feat_dir, 'pt_files', bag_base + '.pt'))
        
        except Exception as e:
            print(f" Error processing {slide_id}: {str(e)}")
            continue

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Process train and test splits
    for split in ['train', 'test']:
        print(f"\n Processing {split.upper()} split")
        print(f"Slide source: {cfg['paths']['slide_dir'][split]}")
        print(f"Patch H5 dir: {cfg['paths']['patch_dir'][split]}")
        print(f"Feature save dir: {os.path.join(cfg['paths']['save_dir'], split, 'features_fp')}")
        process_split(split, cfg)

if __name__ == "__main__":
    main()