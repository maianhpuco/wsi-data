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
from transformers import AutoModel, AutoImageProcessor
from transformers import AutoProcessor, AutoModel
 
# Handle corrupted PNGs
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def quilt_transforms(target_size=224):
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


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


def save_embeddings(model, fname, dataloader, assets_dir):
    processor = AutoProcessor.from_pretrained(
        assets_dir,
        local_files_only=True
    )
    
    if os.path.isfile(f'{fname}.h5'):
        print(f"Already exists: {fname}.h5")
        return

    embeddings, coords = [], []

    for batch, coord in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :]
            print("Features shape: ", feats.shape)
            # feats = model({"pixel_values": batch}).last_hidden_state[:, 0, :]
            # feats = model(batch).last_hidden_state[:, 0, :]  # CLS token
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
    print(f"Extracting Quilt features for: {args.dataset_name}")

    print("Loading Quilt model from:", args.model_name)
    model = AutoModel.from_pretrained(
        args.assets_dir,
        local_files_only=True  # tells HF to load from disk only
    ).to(device).eval()

    # model = AutoModel.from_pretrained(args.assets_dir).to(device).eval()
    transform = quilt_transforms(args.target_patch_size)

    for slide in tqdm(os.listdir(args.patches_path)):
        slide_path = os.path.join(args.patches_path, slide)
        if not os.path.isdir(slide_path):
            continue

        dataset = PatchesDataset(slide_path, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        out_path = os.path.join(args.quilt_features_path, slide)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if not os.path.exists(out_path + ".h5"):
            save_embeddings(model, out_path, dataloader, args.assets_dir)



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
    args.quilt_args = config['quilt_feature_extraction']

    key = f"patch_{args.patch_size}x{args.patch_size}_{args.magnification}"
    if key not in args.paths['patch_png_dir']:
        raise ValueError(f"[✗] Missing key '{key}' in config['paths']['patch_png_dir']")
    args.patches_path = args.paths['patch_png_dir'][key]

    if key not in args.paths['quilt_features_path']:
        raise ValueError(f"[✗] Missing key '{key}' in config['paths']['quilt_features_path']")
    args.quilt_features_path = args.paths['quilt_features_path'][key]
    os.makedirs(args.quilt_features_path, exist_ok=True)

    args.model_name = args.quilt_args['model_name']
    args.assets_dir = args.quilt_args['assets_dir']
    args.batch_size = args.quilt_args['batch_size']
    args.target_patch_size = args.quilt_args.get('target_patch_size', 224)

    print(">>> result will be saved to:", args.quilt_features_path)
    main(args)
