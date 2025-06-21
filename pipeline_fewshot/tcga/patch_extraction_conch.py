import os
import argparse
import yaml
from tqdm import tqdm
import h5py
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import sys

# Add CONCH to path
# Get current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct path to '../../src/externals/CONCH'
_path = os.path.abspath(os.path.join(current_dir, "../../src/externals/CONCH")) 

sys.path.append(_path)
from conch.models.utils import get_transforms
from conch.open_clip_custom import create_model_from_pretrained

# Handle corrupted PNGs
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.multiprocessing.set_sharing_strategy('file_system')


class PatchesDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.imgs = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.png')]
        self.coords = [os.path.basename(f) for f in self.imgs]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        coord = self.coords[index]
        if self.transform:
            img = self.transform(img)
        return img, coord

    def __len__(self):
        return len(self.imgs)


def save_embeddings(model, fname, dataloader):
    if os.path.isfile(f'{fname}.h5'):
        print(f"Already exists: {fname}.h5")
        return

    embeddings, coords = [], []

    for batch, coord in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            feats = model(batch)
            embeddings.append(feats.detach().cpu().numpy())

        for name in coord:
            x, y = map(int, name.replace('.png', '').split('_'))
            coords.append([x, y])

    embeddings = np.vstack(embeddings)
    coords = np.vstack(coords)

    with h5py.File(f'{fname}.h5', 'w') as f:
        f['features'] = embeddings
        f['coords'] = coords


def main(args):
    print(f"Extracting CONCH features for: {args.dataset_name}")

    model, _ = create_model_from_pretrained("conch_ViT-B-16", args.assets_dir)
    from functools import partial
    model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    model = model.to(device).eval()

    transform = get_transforms(args.target_patch_size)

    for slide in tqdm(os.listdir(args.patches_path)):
        slide_path = os.path.join(args.patches_path, slide)
        if not os.path.isdir(slide_path):
            continue

        dataset = PatchesDataset(slide_path, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        out_path = os.path.join(args.conch_features_path, slide)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if not os.path.exists(out_path + ".h5"):
            save_embeddings(model, out_path, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--magnification', type=str, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.dataset_name = config['dataset_name']
    args.paths = config['paths']
    args.conch_args = config['conch_feature_extraction']

    key = f"patch_{args.patch_size}x{args.patch_size}_{args.magnification}"
    if key not in args.paths['patch_png_dir']:
        raise ValueError(f"[✗] Missing key '{key}' in config['paths']['patch_png_dir']")
    args.patches_path = args.paths['patch_png_dir'][key]

    # Use slide-level feature dir for CONCH
    args.conch_features_path = args.paths['conch_features_path'][key] 
    os.makedirs(args.conch_features_path, exist_ok=True)

    print(">>> result will be saved to:", args.conch_features_path)

    args.model_name = args.conch_args['model_name']
    args.batch_size = args.conch_args['batch_size']
    args.assets_dir = args.conch_args['assets_dir']
    args.target_patch_size = args.conch_args['target_patch_size']

    main(args)
