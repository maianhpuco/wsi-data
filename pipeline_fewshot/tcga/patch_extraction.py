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
import clip
import sys

sys.path.append('src/externals/ViLa-MIL')

from feature_extraction.nn_encoder_arch.vision_transformer import vit_small
from feature_extraction.nn_encoder_arch.resnet_trunc import resnet50_trunc_baseline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.multiprocessing.set_sharing_strategy('file_system')
ImageFile.LOAD_TRUNCATED_IMAGES = True


def eval_transforms_clip(pretrained=False):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) if pretrained else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean, std=std)
    ])


def eval_transforms(pretrained=False):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) if pretrained else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


class PatchesDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.imgs = [os.path.join(file_path, f) for f in os.listdir(file_path)]
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


def save_embeddings(model, fname, dataloader, enc_name, overwrite=False):
    if os.path.isfile(f'{fname}.h5') and not overwrite:
        return

    embeddings, coords = [], []

    for batch, coord in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            feats = model(batch) if enc_name not in ['clip_RN50', 'clip_ViTB32'] else model.encode_image(batch)
            embeddings.append(feats.detach().cpu().numpy().squeeze())

        for name in coord:
            x, y = map(int, name.replace('.png', '').split('_'))
            coords.append([x, y])

    embeddings = np.vstack(embeddings)
    coords = np.vstack(coords)

    with h5py.File(f'{fname}.h5', 'w') as f:
        f['features'] = embeddings
        f['coords'] = coords


def main(args):
    print(f"Extracting features for: {args.dataset_name} via {args.model_name}")

    if args.model_name == 'resnet50_trunc':
        model = resnet50_trunc_baseline(pretrained=True)
        transform = eval_transforms(pretrained=True)

    elif args.model_name == 'clip_RN50':
        model, _ = clip.load("RN50", device=device)
        transform = eval_transforms_clip(pretrained=True)

    elif args.model_name == 'clip_ViTB32':
        model, _ = clip.load("ViT-B/32", device=device)
        transform = eval_transforms_clip(pretrained=True)

    elif args.model_name in ['model_dino', 'dino_HIPT']:
        ckpt_path = os.path.join(args.assets_dir, f'{args.model_name}.pth')
        model = vit_small(patch_size=16)
        state_dict = torch.load(ckpt_path, map_location='cpu')['teacher']
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        transform = eval_transforms(pretrained=False)

    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    model = model.to(device)
    if args.model_name not in ['clip_RN50', 'clip_ViTB32']:
        model = torch.nn.DataParallel(model)
    model.eval()

    for slide in tqdm(os.listdir(args.patches_path)):
        slide_path = os.path.join(args.patches_path, slide)
        if not os.path.isdir(slide_path):
            continue

        dataset = PatchesDataset(slide_path, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        out_path = os.path.join(args.clip_rn50_features_path, slide)
        if not os.path.exists(out_path + ".h5"):
            save_embeddings(model, out_path, dataloader, args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--magnification', type=str, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.dataset_name = config['dataset_name']

    # Dynamic patch folder selection
    key = f"patch_{args.patch_size}x{args.patch_size}_{args.magnification}"
    if key not in args.paths:
        raise ValueError(f"[✗] Missing key '{key}' in config['paths']")
    args.patches_path = args.paths[key]

    args.clip_rn50_features_path = args.paths['clip_rn50_features_path']
    args.assets_dir = args.paths.get('assets_dir', './ckpts')
    args.device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.clip_rn50_features_path, exist_ok=True)

    print(" > Start feature extraction for dataset:", args.dataset_name)
    main(args)
