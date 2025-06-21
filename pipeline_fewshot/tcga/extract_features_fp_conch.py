import os
import sys
import argparse
import yaml
import time
import torch
import h5py
import openslide
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from huggingface_hub import login

# === Hugging Face Token and Cache ===
os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/wsi-data/cache_folder/.cache/huggingface'
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("Please set HF_TOKEN in your environment: export HF_TOKEN=your_token")
login(token=hf_token)

# === CLAM and CONCH Setup ===
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_w_loader(output_path, loader, model, verbose=0):
    if verbose > 0:
        print(f'Processing {len(loader)} batches')
    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():
            batch = data['img'].to(device, non_blocking=True)
            coords = data['coord'].numpy().astype(np.int32)

            features = model(batch)
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Feature extraction using CONCH (YAML-driven)')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Paths
    source_dir = cfg['paths']['source_dir']
    patch_h5_dir = cfg['paths']['patch_h5_dir']
    feat_dir = os.path.join(cfg['paths']['save_dir'], 'features_fp')
    slide_name_file = cfg['paths']['slide_name_file']
    uuid_name_file = cfg['paths']['uuid_name_file']
    slide_list_csv = cfg['paths'].get('slide_list', os.path.join(cfg['paths']['save_dir'], 'slide_list.csv'))

    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)

    # Load metadata
    slide_df = pd.read_excel(slide_name_file)
    uuid_df = pd.read_excel(uuid_name_file)
    uuid_map = dict(zip(uuid_df['Filename'], uuid_df['UUID']))
    slide_ext = cfg['conch_feature_extraction'].get('slide_ext', '.svs')

    slide_files = [f for f in slide_df['Filename'].tolist() if f in uuid_map]
    slide_paths = {}
    for fname in slide_files:
        uuid = uuid_map[fname]
        full_path = os.path.join(source_dir, uuid, fname)
        if os.path.exists(full_path):
            slide_paths[fname.replace(slide_ext, '')] = full_path
        else:
            print(f"Missing slide: {full_path}")

    # Write slide list to CSV
    with open(slide_list_csv, 'w') as f:
        f.write("slide_id\n")
        for slide_id in slide_paths:
            f.write(slide_id + slide_ext + '\n')

    # Load processing configs
    patch_size = cfg['processing']['patch_size']
    patch_level = cfg['processing']['patch_level']
    model_name = cfg['conch_feature_extraction']['model_name']
    batch_size = cfg['conch_feature_extraction']['batch_size']
    target_patch_size = cfg['conch_feature_extraction']['target_patch_size']
    no_auto_skip = cfg['conch_feature_extraction'].get('no_auto_skip', False)

    # Init dataset
    bags_dataset = Dataset_All_Bags(slide_list_csv)
    dest_files = os.listdir(os.path.join(feat_dir, 'pt_files'))
    model, img_transforms = get_encoder(model_name, target_img_size=target_patch_size)
    model = model.eval().to(device)

    print(f"🔍 Processing {len(bags_dataset)} slides using {model_name}...")

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == 'cuda' else {}

    for idx in tqdm(range(len(bags_dataset))):
        slide_id = bags_dataset[idx].split(slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(patch_h5_dir, bag_name)
        slide_file_path = slide_paths.get(slide_id)

        if not os.path.exists(h5_file_path):
            print(f"⚠️ Missing H5 file: {h5_file_path}")
            continue
        if not slide_file_path or not os.path.exists(slide_file_path):
            print(f"⚠️ Missing slide file: {slide_file_path}")
            continue
        if not no_auto_skip and slide_id + '.pt' in dest_files:
            print(f"⏩ Skipping {slide_id}")
            continue

        output_path = os.path.join(feat_dir, 'h5_files', bag_name)
        try:
            wsi = openslide.open_slide(slide_file_path)
            dataset = Whole_Slide_Bag_FP(
                file_path=h5_file_path,
                wsi=wsi,
                img_transforms=img_transforms,
                patch_size=patch_size,
                patch_level=patch_level
            )
            if len(dataset) == 0:
                print(f"⚠️ No patches found for {slide_id}")
                continue

            loader = DataLoader(dataset, batch_size=batch_size, **loader_kwargs)
            t_start = time.time()
            compute_w_loader(output_path, loader=loader, model=model, verbose=1)
            duration = time.time() - t_start
            print(f"✅ Done: {slide_id} in {duration:.2f}s")

            with h5py.File(output_path, "r") as f:
                features = f['features'][:]
            torch.save(torch.from_numpy(features),
                       os.path.join(feat_dir, 'pt_files', slide_id + '.pt'))

        except Exception as e:
            print(f"❌ Failed {slide_id}: {e}")
            continue


if __name__ == '__main__':
    main()
