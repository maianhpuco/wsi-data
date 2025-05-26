import os
import glob
import json
import cv2
import tifffile
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# -------------------------- Helper Functions --------------------------
def find_data_group(file_path, data_group):
    is_test = '/test/' in file_path
    file_id = Path(file_path).stem
    for key, ids in data_group.items():
        if str(file_id) in map(str, ids):
            return is_test, key
    return False, 'Unknown'

def process_points(xpoints, ypoints):
    assert len(xpoints) == len(ypoints), "xpoints and ypoints must be the same length"
    new_x = [x for x, y in zip(xpoints, ypoints) if x != 0 and y != 0]
    new_y = [y for x, y in zip(xpoints, ypoints) if x != 0 and y != 0]
    new_x.append(new_x[0])
    new_y.append(new_y[0])
    return np.array(new_x), np.array(new_y)

def get_crop_box(anno, wsi_shape, crop_size):
    H, W = wsi_shape
    xpoints, ypoints = process_points(anno['xpoints'], anno['ypoints'])
    x_c, y_c = (xpoints.min() + xpoints.max()) // 2, (ypoints.min() + ypoints.max()) // 2
    x_min, x_max = max(0, x_c - crop_size // 2), min(W, x_c + crop_size // 2)
    y_min, y_max = max(0, y_c - crop_size // 2), min(H, y_c + crop_size // 2)
    return x_min, y_min, x_max, y_max

def count_labels(df):
    return dict(df['label'].value_counts())

# -------------------------- Main Processing Function --------------------------
def process_annotations(data_root, ann_root, crop_size, save_dir):
    data_group = {
        'FastRed_Mouse': ['6533664', '6533666', '6533668', '6533679', '6533672', '6533678', '6533683', '6533680', '6533682', '6533691', '6533687', '6533688'],
        'H_E_Mouse_G1': ['6654562', '6654559', '6654566', '6654568', '6654582', '6654586', '6654587', '6654588', '6654517'],
        'H_E_Mouse_G2': ['6479221', '6479222', '6479223', '6479224', '6479183', '6479186', '6479191', '6479195', '6479196', '6479203'],
        'H_E_Mouse_G3': ['6666707', '6666708', '6666709', '6666710', '6666711', '6666712', '6666713', '6666714', '6666717', '6666718'],
        'PAS_Rat': ['6609616', '6609615', '6609629', '6609617', '6609626', '6609628', '6625497', '6625501', '6625506', '6625505', '6609613', '6609634', '6609605'],
        'H_DAB_Rat_G1': ['5483162', '5482449', '5483170', '5482455', '5483190', '5482452', '5482458', '5483139', '5483117', '5482411', '5483132'],
        'H_DAB_Rat_G2': ['4737452', '4737489', '4737509', '4737522', '4730025', '4730043', '4730080', '4758050', '4758056', '4758065', '4758073'],
        'H_DAB_Rat_G3': ['6139966', '6139967', '6139977', '6139983', '6140227', '6140234', '6140254', '6140232', '6140251', '6140179', '6140168', '6140176'],
    }

    exclude_wsis = {'6666717', '6533672', '6666713', '6533682', '6666711'}
    extracted_dir = Path(data_root) / 'extracted_data'
    seg_ann_paths = sorted(glob.glob(str(Path(ann_root) / '*.json')))
    print(f"Found {len(seg_ann_paths)} JSON annotation files.")

    record = []

    for ann_path in tqdm(seg_ann_paths, desc="Processing annotations"):
        with open(ann_path, 'r') as f:
            data = json.load(f)

        seg_data = [a for a in data['seg_data'] if a.get('sub_type') == 'Normal']
        wsi_path = os.path.join(data_root, data['data_path'].lstrip('/'))
        if not os.path.exists(wsi_path):
            print(f"⚠️ Missing WSI: {wsi_path}")
            continue

        wsi_id = Path(wsi_path).stem
        wsi_img = tifffile.imread(wsi_path, key=0)
        H, W, _ = wsi_img.shape

        is_test, label = find_data_group(wsi_path, data_group)
        if label == 'Unknown':
            continue

        split = 'test' if is_test else 'train'
        save_mask_dir = extracted_dir / f"{split}_mask"
        save_img_root = extracted_dir / f"{'Testing' if is_test else 'Training'}_data_patch" / label / wsi_id / 'img'
        save_mask_root = extracted_dir / f"{'Testing' if is_test else 'Training'}_data_patch" / label / wsi_id / 'mask'
        os.makedirs(save_img_root, exist_ok=True)
        os.makedirs(save_mask_root, exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)

        # Create global mask
        all_areas = [np.array(list(zip(*process_points(a['xpoints'], a['ypoints'])))) for a in seg_data]
        mask_data = cv2.fillPoly(np.zeros((H, W), dtype=np.uint8), all_areas, 1)

        for count, anno in enumerate(seg_data, 1):
            x_min, y_min, x_max, y_max = get_crop_box(anno, (H, W), crop_size)
            crop_img = cv2.cvtColor(wsi_img[y_min:y_max, x_min:x_max], cv2.COLOR_RGB2BGR)
            crop_mask = mask_data[y_min:y_max, x_min:x_max]

            fname = f'{label}_{wsi_id}_{count}_{x_min}_{y_min}'
            img_path = save_img_root / f'{fname}_img.jpg'
            mask_path = save_mask_root / f'{fname}_mask.png'
            cv2.imwrite(str(img_path), crop_img)
            cv2.imwrite(str(mask_path), crop_mask * 255)

            record.append({'filename': fname, 'wsi_id': wsi_id, 'group': label, 'split': split, 'img': str(img_path), 'mask': str(mask_path), 'label': label})

        if not is_test and wsi_id not in exclude_wsis:
            tifffile.imwrite(str(save_mask_dir / f'{wsi_id}.tiff'), mask_data)

    df = pd.DataFrame(record)
    return df, extracted_dir

# -------------------------- Cross-Validation Split --------------------------
def create_folds(df, save_dir, n_splits=5):
    os.makedirs(save_dir, exist_ok=True)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        fold_dir = Path(save_dir) / f'fold_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        fold_train.to_csv(fold_dir / 'train.csv', index=False)
        fold_val.to_csv(fold_dir / 'val.csv', index=False)
        test_df.to_csv(fold_dir / 'test.csv', index=False)

        print(f"\nFold {fold}")
        print(f"Train: {len(fold_train)} samples -> {count_labels(fold_train)}")
        print(f"Val:   {len(fold_val)} samples -> {count_labels(fold_val)}")
        print(f"Test:  {len(test_df)} samples -> {count_labels(test_df)}")
        print(f"Total: {len(fold_train) + len(fold_val) + len(test_df)} / {len(df)}")

# -------------------------- Execution --------------------------
# DATA_ROOT = "/datasets/glomeruli"
# ANN_ROOT = "/datasets/glomeruli/annotations"
# CROP_SIZE = 1024

df, extracted_patch_dir = process_annotations(DATA_ROOT, ANN_ROOT, CROP_SIZE, save_dir=DATA_ROOT)
create_folds(df, save_dir=extracted_patch_dir / "splits", n_splits=5)
