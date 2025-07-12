import os
import sys
import time
import yaml
import torch
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import ImageFile
import openslide
from torch.utils.data import DataLoader
from torchvision import transforms

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def eval_transforms_clip(pretrained=True):
    mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if pretrained else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean, std=std)
    ])

def compute_w_loader(output_path, loader, model):
    mode = 'w'
    for data in tqdm(loader):
        with torch.inference_mode():
            imgs = data['img'].to(device)
            coords = data['coord'].numpy().astype(np.int32)
            feats = model(imgs).cpu().numpy().astype(np.float32)

        save_hdf5(output_path, {'features': feats, 'coords': coords}, attr_dict=None, mode=mode)
        mode = 'a'
    return output_path

def main(args):
    cfg = load_config(args.config)
    conch_cfg = cfg['conch_feature_extraction']

    # === Load config paths ===
    source_dir = cfg['paths']['source_dir']
    # patch_h5_dir = cfg['paths']['patch_h5_dir']
    if args.magnification == '5x': 
        patch_h5_dir = cfg['paths']['patch_h5_dir_5x']
    elif args.magnification == '10x':
        patch_h5_dir = cfg['paths']['patch_h5_dir_10x']  
        
    feat_dir = cfg['paths']['conch_features_path'][f"patch_{args.patch_size}x{args.patch_size}_{args.magnification}"]
    os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)

    slide_name_file = cfg['paths']['slide_name_file']
    uuid_name_file = cfg['paths']['uuid_name_file']
    csv_path = cfg['paths'].get('slide_list', os.path.join(cfg['paths']['save_dir'], 'slide_list.csv'))

    # === Read slide name list and UUIDs ===
    slide_df = pd.read_excel(slide_name_file)
    uuid_df = pd.read_excel(uuid_name_file)
    uuid_map = dict(zip(uuid_df['Filename'], uuid_df['UUID']))

    slide_ext = conch_cfg.get("slide_ext", ".svs")
    slide_files = [s for s in slide_df['Filename'] if s in uuid_map]

    slide_paths = {}
    for s in slide_files:
        uuid = uuid_map[s]
        full_path = os.path.join(source_dir, uuid, s)
        if os.path.exists(full_path):
            slide_paths[s.replace(slide_ext, '')] = full_path

    if not slide_paths:
        print("No valid slide paths found.")
        return

    with open(csv_path, 'w') as f:
        f.write("slide_id\n")
        for sid in slide_paths.keys():
            f.write(sid + slide_ext + '\n')

    # === Load model ===
    model, _ = create_model_from_pretrained(conch_cfg['model_name'], conch_cfg['assets_dir'])
    model.forward = lambda x: model.encode_image(x, proj_contrast=False, normalize=False)
    model = model.to(device).eval()
    img_transform = eval_transforms_clip()

    dataset = Dataset_All_Bags(csv_path)
    saved = os.listdir(os.path.join(feat_dir, 'pt_files'))
    batch_size = conch_cfg.get("batch_size", 64)
    no_auto_skip = conch_cfg.get("no_auto_skip", False)
    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

    for idx in tqdm(range(len(dataset))):
        slide_id = dataset[idx].split(slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_path = os.path.join(patch_h5_dir, bag_name)
        slide_path = slide_paths.get(slide_id)

        if not os.path.exists(h5_path) or not slide_path:
            print(f"[!] Missing file for {slide_id}")
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
            compute_w_loader(output_path, loader, model)

            with h5py.File(output_path, "r") as f:
                features = torch.from_numpy(f['features'][:])
            torch.save(features, os.path.join(feat_dir, 'pt_files', slide_id + ".pt"))

        except Exception as e:
            print(f"[✗] Error on {slide_id}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CONCH feature extraction for TCGA KICH/KIRC/KIRP")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--magnification', type=str, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    args = parser.parse_args()
    main(args)
