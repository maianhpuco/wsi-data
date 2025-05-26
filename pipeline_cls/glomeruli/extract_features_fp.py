import os
import sys
import argparse
import yaml
import time
import torch
import h5py
import openslide
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_w_loader(output_path, loader, model, verbose=0):
    if verbose > 0:
        print(f'Processing a total of {len(loader)} batches')

    mode = 'w'
    for data in tqdm(loader):
        with torch.inference_mode():
            batch = data['img'].to(device, non_blocking=True)
            coords = data['coord'].numpy().astype(np.int32)
            features = model(batch).cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
    return output_path

def process_split(split, cfg):
    source = cfg['paths']['slide_dir'][split]
    patch_h5_dir = cfg['paths']['patch_dir'][split]
    feat_dir = os.path.join(cfg['paths']['save_dir'], 'features_fp', split)
    csv_path = os.path.join(cfg['paths']['save_dir'], split, 'slide_list.csv')

    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)

    print(f"Generating slide list CSV for {split} at: {csv_path}")
    slide_ext = cfg['feature_extraction'].get("slide_ext", ".tif")
    slide_files = [f for f in os.listdir(source) if f.endswith(slide_ext)]
    if not slide_files:
        print(f"No slides found in {source} with extension {slide_ext}")
        return

    with open(csv_path, 'w') as f:
        f.write("slide_id\n")
        for s in slide_files:
            f.write(s + '\n')

    model_name = cfg['feature_extraction'].get("model_name", "resnet50_trunc")
    batch_size = cfg['feature_extraction'].get("batch_size", 256)
    target_patch_size = cfg['feature_extraction'].get("target_patch_size", 224)
    no_auto_skip = cfg['feature_extraction'].get("no_auto_skip", False)

    model, img_transforms = get_encoder(model_name, target_img_size=target_patch_size)
    model.eval().to(device)

    bags_dataset = Dataset_All_Bags(csv_path)
    if len(bags_dataset) == 0:
        print(f"No slides found in dataset from {csv_path}")
        return

    dest_files = os.listdir(os.path.join(feat_dir, 'pt_files'))
    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    for idx in tqdm(range(len(bags_dataset)), desc=f"Processing {split}"):
        slide_id = bags_dataset[idx].split(slide_ext)[0]
        h5_file = os.path.join(patch_h5_dir, f"{slide_id}.h5")
        slide_file = os.path.join(source, f"{slide_id}{slide_ext}")

        if not no_auto_skip and f"{slide_id}.pt" in dest_files:
            print(f"Skipped {slide_id} (features already exist)")
            continue

        if not os.path.exists(h5_file) or not os.path.exists(slide_file):
            print(f"Missing file: {h5_file if not os.path.exists(h5_file) else slide_file}")
            continue

        output_path = os.path.join(feat_dir, 'h5_files', f"{slide_id}.h5")
        try:
            wsi = openslide.open_slide(slide_file)
            dataset = Whole_Slide_Bag_FP(file_path=h5_file, wsi=wsi, img_transforms=img_transforms)
            if len(dataset) == 0:
                print(f"No patches found in {h5_file}")
                continue

            loader = DataLoader(dataset, batch_size=batch_size, **loader_kwargs)
            compute_w_loader(output_path, loader, model, verbose=1)

            with h5py.File(output_path, "r") as f:
                features = torch.from_numpy(f['features'][:])
            torch.save(features, os.path.join(feat_dir, 'pt_files', f"{slide_id}.pt"))

        except Exception as e:
            print(f"Error processing {slide_id}: {e}")
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
        print(f"Feature save dir: {os.path.join(cfg['paths']['save_dir'],'features_fp', split)}")
        process_split(split, cfg)

if __name__ == "__main__":
    main()