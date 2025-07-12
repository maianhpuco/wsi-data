import os
import sys
import time
import yaml
import torch
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
from torch.utils.data import DataLoader
import openslide
# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setup paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "src/externals/CONCH"))

# Imports
from conch.open_clip_custom import create_model_from_pretrained
from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torchvision import transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def eval_transforms_clip(pretrained=True):
    mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if pretrained else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean, std=std)
    ])

def compute_w_loader(output_path, loader, model, verbose=0):
    if verbose > 0:
        print(f"[INFO] Processing {len(loader)} batches")
    mode = 'w'
    for data in tqdm(loader):
        with torch.inference_mode():
            imgs = data['img'].to(device)
            coords = data['coord'].numpy().astype(np.int32)
            feats = model(imgs).cpu().numpy().astype(np.float32)

        asset_dict = {'features': feats, 'coords': coords}
        save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
        mode = 'a'
    return output_path

def main(args):
    cfg = load_config(args.config)
    conch_cfg = cfg['conch_feature_extraction']
    key = f"patch_{args.patch_size}x{args.patch_size}_{args.magnification}"

    # Paths
    source = cfg['paths']['source']
    
    if args.magnification == '5x': 
        patch_h5_dir = cfg['paths']['patch_save_dir_5x']
    elif args.magnification == '10x':
        patch_h5_dir = cfg['paths']['patch_save_dir_10x'] 
        
    feat_dir = cfg['paths']['conch_features_path'][key]
    os.makedirs(feat_dir, exist_ok=True)
    csv_path = os.path.join(cfg['paths']['save_dir'], 'slide_list.csv')

    os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)

    # Generate CSV
    slide_ext = conch_cfg.get("slide_ext", ".tiff")
    slide_files = [f for f in os.listdir(source) if f.endswith(slide_ext)]
    if not slide_files:
        print(f"[✗] No slides found in {source}")
        sys.exit(1)
    with open(csv_path, 'w') as f:
        f.write("slide_id\n")
        for slide in slide_files:
            f.write(slide + "\n")

    # Init dataset
    dataset = Dataset_All_Bags(csv_path)
    if len(dataset) == 0:
        print(f"[✗] No slides listed in {csv_path}")
        sys.exit(1)

    # Load model
    model_name = conch_cfg["model_name"]
    assets_dir = conch_cfg["assets_dir"]
    batch_size = conch_cfg.get("batch_size", 64)
    target_patch_size = conch_cfg.get("target_patch_size", 224)
    no_auto_skip = conch_cfg.get("no_auto_skip", False)

    model, _ = create_model_from_pretrained(model_name, assets_dir)
    model.forward = lambda x: model.encode_image(x, proj_contrast=False, normalize=False)
    model = model.to(device).eval()
    img_transform = eval_transforms_clip()

    # DataLoader settings
    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == 'cuda' else {}
    saved = os.listdir(os.path.join(feat_dir, 'pt_files'))

    for idx in tqdm(range(len(dataset))):
        slide_id = dataset[idx].split(slide_ext)[0]
        bag_name = slide_id + ".h5"
        h5_path = os.path.join(patch_h5_dir, bag_name)
        slide_path = os.path.join(source, slide_id + slide_ext)

        print(f"\n[{idx+1}/{len(dataset)}] Slide: {slide_id}")
        if not os.path.exists(h5_path) or not os.path.exists(slide_path):
            print(f"[!] Missing files for: {slide_id}")
            continue

        if not no_auto_skip and slide_id + ".pt" in saved:
            print(f"[✓] Skipped {slide_id}")
            continue

        try:
            output_path = os.path.join(feat_dir, 'h5_files', bag_name)
            
            wsi = openslide.open_slide(slide_path)
            dset = Whole_Slide_Bag_FP(file_path=h5_path, wsi=wsi, img_transforms=img_transform)
            if len(dset) == 0:
                print(f"[!] No patches in {bag_name}")
                continue

            loader = DataLoader(dataset=dset, batch_size=batch_size, **loader_kwargs)
            compute_w_loader(output_path, loader=loader, model=model, verbose=1)

            # Save to .pt
            with h5py.File(output_path, "r") as f:
                features = torch.from_numpy(f['features'][:])
            torch.save(features, os.path.join(feat_dir, 'pt_files', slide_id + ".pt"))

        except Exception as e:
            print(f"[✗] Failed {slide_id}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CONCH Feature Extraction for WSIs")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--magnification', type=str, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    args = parser.parse_args()

    main(args)
