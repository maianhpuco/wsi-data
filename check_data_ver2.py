import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import defaultdict

# === PATH SETUP ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'src')))

def prepare_dataset(args, fold_id):
    data_dir_map = args.data_dir_map_config
    data_dir_mapping = args.paths[data_dir_map] 

    if args.dataset_name in ['tcga_renal', 'tcga_lung']:
        from datasets.classification.tcga import return_splits_custom 
        train_dataset, val_dataset, test_dataset = return_splits_custom(
            train_csv_path=os.path.join(args.paths['split_folder'], f"fold_{fold_id}/train.csv"),
            val_csv_path=os.path.join(args.paths['split_folder'], f"fold_{fold_id}/val.csv"),
            test_csv_path=os.path.join(args.paths['split_folder'], f"fold_{fold_id}/test.csv"), 
            data_dir_map=data_dir_mapping, 
            label_dict=args.label_dict, 
            seed=1, 
            print_info=False, 
            use_h5=False
        ) 
        print(len(train_dataset))
        return train_dataset, val_dataset, test_dataset  

    elif args.dataset_name == 'camelyon16':
        from datasets.classification.camelyon16 import return_splits_custom
        split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        train_dataset, val_dataset, test_dataset = return_splits_custom(
            csv_path=split_csv_path,
            data_dir=data_dir_mapping,
            label_dict=args.label_dict,
            seed=1,
            print_info=True,
            use_h5=getattr(args, 'use_h5', False)
        )
        print(f"[INFO] Loaded {len(train_dataset)} train samples")
        return train_dataset, val_dataset, test_dataset

    else:
        raise NotImplementedError(f"[✗] Dataset '{args.dataset_name}' not supported.")

def check_data(fold_id, data_dir_map_config, args):
    split_dir = args.paths['split_folder']

    if args.dataset_name == 'camelyon16':
        split_csv_path = os.path.join(split_dir, f"fold_{fold_id}.csv")
        df = pd.read_csv(split_csv_path)
        dfs = []
        for split in ["train", "val", "test"]:
            dfs.append(pd.DataFrame({
                "slide": df[split],
                "label": df[f"{split}_label"],
                "split": split
            }))
        df_full = pd.concat(dfs, ignore_index=True).dropna()
        df_full['label'] = df_full['label'].astype(int)
    else:
        train_csv_path = os.path.join(split_dir, f"fold_{fold_id}/train.csv")
        val_csv_path   = os.path.join(split_dir, f"fold_{fold_id}/val.csv")
        test_csv_path  = os.path.join(split_dir, f"fold_{fold_id}/test.csv")

        train_df = pd.read_csv(train_csv_path)
        val_df   = pd.read_csv(val_csv_path)
        test_df  = pd.read_csv(test_csv_path)

        train_df['split'] = 'train'
        val_df['split']   = 'val'
        test_df['split']  = 'test'

        df_full = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"Total samples: {len(df_full)}")

    log_dir = os.path.join("/project/hnguyen2/mvu9/folder_04_ma/logs", "missing_tcga", args.dataset_name, data_dir_map_config)
    os.makedirs(log_dir, exist_ok=True)

    data_dir_map = args.paths[data_dir_map_config]
    label_counts = defaultdict(lambda: {"available": 0, "missing": 0})

    for label in df_full['label'].unique():
        label_str = str(label).lower()
        df_label = df_full[df_full['label'] == label]
        missing = []

        for _, row in df_label.iterrows():
            slide_id = row['slide']
            slide_path = os.path.join(data_dir_map[label_str], 'h5_files', f"{slide_id}.h5")

            if os.path.exists(slide_path):
                label_counts[label]["available"] += 1
            else:
                label_counts[label]["missing"] += 1
                missing.append((slide_id,))

        if missing:
            df_missing = pd.DataFrame(missing, columns=['slide_id'])
            save_path = os.path.join(log_dir, f"missing_{label_str}.csv")
            # df_missing.to_csv(save_path, index=False)

    total_counts = {label: counts["available"] + counts["missing"] for label, counts in label_counts.items()}
    df_summary = pd.DataFrame([
        {
            "label": label,
            "slide_available": counts["available"],
            "slide_missing": counts["missing"],
            "%_available": f"{100 * counts['available'] / total_counts[label]:.2f}%",
            "%_missing": f"{100 * counts['missing'] / total_counts[label]:.2f}%"
        }
        for label, counts in label_counts.items()
    ])

    print(df_summary)
    return df_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--k_start', type=int, required=True)
    parser.add_argument('--k_end', type=int, required=True)
    parser.add_argument('--data_dir_map', type=str, default=None, help='Path to the data directory mapping file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("################# SETTINGS ###################")
    for k, v in vars(args).items():
        if k != 'paths':
            print(f"{k}: {v}")
    print("##############################################")

    for fold_id in range(args.k_start, args.k_end + 1):
        print(f"Processing fold {fold_id}...")
        data_dir_map_configs = [
            'conch_patch_256x256_5x', 
            'conch_patch_256x256_10x', 
            'clip_rn50_patch_256x256_5x',
            'clip_rn50_patch_256x256_10x',
        ]
        for data_dir_map_config in data_dir_map_configs:  
            print(f">> Checking data for {data_dir_map_config}...")
            check_data(fold_id, data_dir_map_config, args)
            print("---------------------------------------")
