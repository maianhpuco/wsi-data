import os
import sys
import argparse
import yaml
import time
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import h5py
import openslide
from tqdm import tqdm
import numpy as np
import pandas as pd

# Ensure CLAM is in the import path
# sys.path.append("src/externals/CLAM")
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) 
sys.path.append(base_path)
from utils.file_utils import save_hdf5
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

# Check CUDA availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print(f"CUDA is available. Using: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Using CPU only.")

def load_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load config file {config_path}: {str(e)}")

def compute_w_loader(output_path, loader, model, verbose=0):
    """Compute features for a DataLoader and save to HDF5."""
    if verbose > 0:
        print(f'Processing a total of {len(loader)} batches')

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        try:
            with torch.inference_mode():
                batch = data['img']
                coords = data['coord'].numpy().astype(np.int32)
                batch = batch.to(device, non_blocking=True)
                
                features = model(batch)
                features = features.cpu().numpy().astype(np.float32)

                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'
        except Exception as e:
            print(f"Error processing batch {count}: {str(e)}")
            continue
    
    return output_path

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Set paths from config
    source = cfg['paths']['source_dir']
    patch_h5_dir = cfg['paths']['patch_h5_dir']
    feat_dir = os.path.join(cfg['paths']['save_dir'], 'features_fp')
    slide_name_file = cfg['paths']['slide_name_file']
    uuid_name_file = cfg['paths'].get('uuid_name_file')
    csv_path = cfg['paths'].get('slide_list', os.path.join(cfg['paths']['save_dir'], 'slide_list.csv'))

    # Create necessary directories
    try:
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
        os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        sys.exit(1)

    # Read slide list from Excel
    print(f"🔧 Reading slide list from: {slide_name_file}")
    try:
        slide_df = pd.read_excel(slide_name_file)
        if 'Filename' not in slide_df.columns:
            print(f"❌ 'Filename' column not found in {slide_name_file}. Available columns: {slide_df.columns.tolist()}")
            sys.exit(1)
        slide_files = slide_df['Filename'].tolist()
    except Exception as e:
        print(f"❌ Error reading {slide_name_file}: {str(e)}")
        sys.exit(1)

    # Read UUID mapping
    uuid_map = {}
    try:
        uuid_df = pd.read_excel(uuid_name_file)
        if 'Filename' not in uuid_df.columns or 'UUID' not in uuid_df.columns:
            print(f"❌ 'Filename' or 'UUID' column not found in {uuid_name_file}. Available columns: {uuid_df.columns.tolist()}")
            sys.exit(1)
        uuid_map = dict(zip(uuid_df['Filename'], uuid_df['UUID']))
    except Exception as e:
        print(f"❌ Error reading {uuid_name_file}: {str(e)}")
        sys.exit(1)

    slide_ext = cfg.get("feature_extraction", {}).get("slide_ext", ".svs")
    print(f"Slide extension: {slide_ext}")

    # Filter slides to those present in both slides.xlsx and uuids.xlsx
    slide_files = [f for f in slide_files if f in uuid_map]
    if not slide_files:
        print(f"❌ No slides found in {slide_name_file} that match {uuid_name_file}")
        sys.exit(1)

    # Find slide paths using UUID mapping
    slide_paths = {}
    for slide_id in slide_files:
        uuid = uuid_map.get(slide_id)
        if uuid:
            slide_path = os.path.join(source, uuid, slide_id)
            if os.path.exists(slide_path):
                slide_paths[slide_id.replace(slide_ext, '')] = slide_path
            else:
                print(f"❌ Slide file not found: {slide_path}")
        else:
            print(f"⚠️ UUID not found for {slide_id}")

    if not slide_paths:
        print(f"❌ No valid slide files found in {source}")
        sys.exit(1)
    print(f"Found {len(slide_paths)} slides: {list(slide_paths.keys())[:5]}")

    # Write slide list to CSV for Dataset_All_Bags
    try:
        with open(csv_path, 'w') as f:
            f.write("slide_id\n")
            for slide_base in slide_paths.keys():
                f.write(slide_base + slide_ext + '\n')
    except Exception as e:
        print(f"❌ Error writing {csv_path}: {str(e)}")
        sys.exit(1)

    # Feature extraction config
    feat_cfg = cfg.get("feature_extraction", {})
    model_name = feat_cfg.get("model_name", "resnet50_trunc")
    batch_size = feat_cfg.get("batch_size", 256)
    target_patch_size = feat_cfg.get("target_patch_size", 224)
    slide_ext = feat_cfg.get("slide_ext", ".svs")
    no_auto_skip = feat_cfg.get("no_auto_skip", False)

    # Processing config
    patch_size = cfg['processing']['patch_size']
    patch_level = cfg['processing']['patch_level']

    # Initialize dataset
    print('Initializing dataset')
    try:
        bags_dataset = Dataset_All_Bags(csv_path)
    except Exception as e:
        print(f"❌ Error initializing dataset: {str(e)}")
        sys.exit(1)

    total = len(bags_dataset)
    if total == 0:
        print(f"❌ No slides found in dataset from {csv_path}")
        sys.exit(1)
    print(f"Total slides in dataset: {total}")

    # Load model
    try:
        model, img_transforms = get_encoder(model_name, target_img_size=target_patch_size)
        model.eval()
        model = model.to(device)
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {str(e)}")
        sys.exit(1)

    # Check for existing feature files
    try:
        dest_files = os.listdir(os.path.join(feat_dir, 'pt_files'))
    except Exception as e:
        print(f"Error accessing pt_files directory: {str(e)}")
        dest_files = []

    # DataLoader kwargs
    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    # Process each slide
    for bag_candidate_idx in tqdm(range(total)):
        slide_id = bags_dataset[bag_candidate_idx].split(slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(patch_h5_dir, bag_name)
        slide_file_path = slide_paths.get(slide_id)
        print(f'\nProgress: {bag_candidate_idx+1}/{total}')
        print(f"Slide ID: {slide_id}")

        # Skip if features already exist and no_auto_skip is False
        if not no_auto_skip and slide_id + '.pt' in dest_files:
            print(f"Skipped {slide_id} (features already exist)")
            continue

        # Verify H5 and slide file existence
        if not os.path.exists(h5_file_path):
            print(f"❌ H5 file not found: {h5_file_path}")
            continue
        if not slide_file_path or not os.path.exists(slide_file_path):
            print(f"❌ Slide file not found: {slide_file_path or 'Unknown'}")
            continue

        # Process slide
        output_path = os.path.join(feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        try:
            wsi = openslide.open_slide(slide_file_path)
            dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
                                        wsi=wsi, 
                                        img_transforms=img_transforms, 
                                        patch_level=patch_level, 
                                        patch_size=patch_size)
            
            if len(dataset) == 0:
                print(f"❌ No patches found in {h5_file_path}")
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
            print(f"❌ Error processing {slide_id}: {str(e)}")
            continue

if __name__ == "__main__":
    main()