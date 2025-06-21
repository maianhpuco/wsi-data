import os
import sys
import argparse
import yaml
import time
import torch
import h5py
import openslide
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add CONCH and CLAM paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "src/externals/CONCH"))


from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from conch.models.encoder import CONCHEncoder
from conch.models.utils import get_transforms
from huggingface_hub import login

# from models import get_encoder 

# from conch.open_clip_custom import create_model_from_pretrained 
# model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
#     model.forward = partial(model.encode_image, proj_contrast=False, normalize=False) 

# Set custom Hugging Face cache directory
os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/wsi-data/cache_folder/.cache/huggingface'

# Load Hugging Face token from environment and login
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("Please set HF_TOKEN in your environment (export HF_TOKEN=xxx)")

login(token=hf_token) 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_w_loader(output_path, loader, model, verbose=0):
    if verbose > 0:
        print(f'Processing {len(loader)} batches')
    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        try:
            with torch.inference_mode():
                batch = data['img'].to(device, non_blocking=True)
                coords = data['coord'].numpy().astype(np.int32)

                features = model(batch)
                features = features.cpu().numpy().astype(np.float32)

                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'
        except Exception as e:
            print(f"Error at batch {count}: {e}")
            continue
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Feature Extraction using CONCH')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Paths
    source = cfg['paths']['source_dir']
    patch_h5_dir = cfg['paths']['patch_h5_dir']
    feat_dir = os.path.join(cfg['paths']['save_dir'], 'features_fp')
    slide_name_file = cfg['paths']['slide_name_file']
    uuid_name_file = cfg['paths']['uuid_name_file']
    csv_path = cfg['paths'].get('slide_list', os.path.join(cfg['paths']['save_dir'], 'slide_list.csv'))

    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)

    # Slide name & UUID
    slide_df = pd.read_excel(slide_name_file)
    uuid_df = pd.read_excel(uuid_name_file)
    uuid_map = dict(zip(uuid_df['Filename'], uuid_df['UUID']))

    slide_ext = cfg['conch_feature_extraction'].get('slide_ext', '.svs')
    slide_files = slide_df['Filename'].tolist()
    slide_files = [f for f in slide_files if f in uuid_map]

    slide_paths = {}
    for fname in slide_files:
        uuid = uuid_map[fname]
        path = os.path.join(source, uuid, fname)
        if os.path.exists(path):
            slide_paths[fname.replace(slide_ext, '')] = path
        else:
            print(f"Missing: {path}")

    # Write slide list CSV
    with open(csv_path, 'w') as f:
        f.write("slide_id\n")
        for k in slide_paths:
            f.write(k + slide_ext + '\n')

    # Config values
    patch_size = cfg['processing']['patch_size']
    patch_level = cfg['processing']['patch_level']
    model_name = cfg['conch_feature_extraction'].get('model_name', 'conch_v1')
    batch_size = cfg['conch_feature_extraction'].get('batch_size', 128)
    target_patch_size = cfg['conch_feature_extraction'].get('target_patch_size', 224)
    no_auto_skip = cfg['conch_feature_extraction'].get('no_auto_skip', False)

    bags_dataset = Dataset_All_Bags(csv_path)
    total = len(bags_dataset)

    print(f"Loaded {total} slides")

    # Load CONCH model
    print("Loading CONCH model...")
    model = CONCHEncoder(pretrained=True).eval().to(device)
    img_transforms = get_transforms(target_patch_size)

    dest_files = os.listdir(os.path.join(feat_dir, 'pt_files'))
    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    for idx in tqdm(range(total)):
        slide_id = bags_dataset[idx].split(slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(patch_h5_dir, bag_name)
        slide_file_path = slide_paths.get(slide_id)

        print(f"\n[{idx+1}/{total}] Slide: {slide_id}")

        if not no_auto_skip and slide_id + '.pt' in dest_files:
            print(f"Skipped {slide_id}")
            continue
        if not os.path.exists(h5_file_path) or not slide_file_path or not os.path.exists(slide_file_path):
            print(f"Missing input for {slide_id}")
            continue

        output_path = os.path.join(feat_dir, 'h5_files', bag_name)
        start = time.time()

        try:
            wsi = openslide.open_slide(slide_file_path)
            dataset = Whole_Slide_Bag_FP(
                file_path=h5_file_path,
                wsi=wsi,
                img_transforms=img_transforms,
                patch_level=patch_level,
                patch_size=patch_size
            )
            if len(dataset) == 0:
                print(f"No patches found in {h5_file_path}")
                continue

            loader = DataLoader(dataset, batch_size=batch_size, **loader_kwargs)
            output_file_path = compute_w_loader(output_path, loader=loader, model=model, verbose=1)

            duration = time.time() - start
            print(f"Feature extraction for {slide_id} took {duration:.2f}s")

            with h5py.File(output_file_path, "r") as f:
                features = f['features'][:]
                print("Feature shape:", features.shape)

            torch.save(torch.from_numpy(features), os.path.join(feat_dir, 'pt_files', slide_id + '.pt'))

        except Exception as e:
            print(f"Failed on {slide_id}: {e}")
            continue

if __name__ == '__main__':
    main()
