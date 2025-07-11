import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime


# === PATH SETUP ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'src')))

def prepare_dataset(args, fold_id):
    if args.dataset_name == 'tcga_renal':
        from datasets.single_scale.tcga import return_splits_custom
        
        patch_size = args.patch_size
        data_dir_map = args.data_dir_map_config
        data_dir_mapping= args.paths[data_dir_map]
    
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
    else:
        raise NotImplementedError(f"[✗] Dataset '{args.dataset_name}' not supported.")
    
def check_data(fold_id, data_dir_map_config, args):
    import pandas as pd
    import os
    from collections import defaultdict

    # Paths to CSVs
    split_dir = args.paths['split_folder']
    train_csv_path = os.path.join(split_dir, f"fold_{fold_id}/train.csv")
    val_csv_path   = os.path.join(split_dir, f"fold_{fold_id}/val.csv")
    test_csv_path  = os.path.join(split_dir, f"fold_{fold_id}/test.csv")

    # Load CSVs
    train_df = pd.read_csv(train_csv_path)
    val_df   = pd.read_csv(val_csv_path)
    test_df  = pd.read_csv(test_csv_path)

    # Tag split
    train_df['split'] = 'train'
    val_df['split']   = 'val'
    test_df['split']  = 'test'

    # Combine all
    df_full = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # print(df_full.head())
    print(f"Total samples: {len(df_full)}")

    # Prepare output folder
    os.makedirs("logs", exist_ok=True)

    # Get path map for .h5 files
    data_dir_map = args.paths[data_dir_map_config]

    # Count available and missing slides per label
    label_counts = defaultdict(lambda: {"available": 0, "missing": 0})
    missing_records = {}

    for label in df_full['label'].unique():
        label_lower = label.lower()
        df_label = df_full[df_full['label'] == label]

        missing = []

        for _, row in df_label.iterrows():
            slide_id = row['slide']
            patient_id = row['patient_id']
            slide_path = os.path.join(data_dir_map[label_lower], f"{slide_id}.h5")

            if os.path.exists(slide_path):
                label_counts[label]["available"] += 1
            else:
                label_counts[label]["missing"] += 1
                missing.append((patient_id, slide_id))

        # Save per-label missing slide info
        if missing:
            df_missing = pd.DataFrame(missing, columns=['patient_id', 'slide_id'])
            missing_records[label_lower] = df_missing
            save_path = f"logs/{data_dir_map_config}/missing_{label_lower}.csv"
            df_missing.to_csv(save_path, index=False)
            
            print(f"[INFO] Saved missing file list for {label} → {save_path}")
        else:
            print(f"[INFO] No missing slides for label: {label}")

    # Save availability summary
    df_summary = pd.DataFrame([
        {"label": label, "slide_available": counts["available"], "slide_missing": counts["missing"]}
        for label, counts in label_counts.items()
    ])
    print("\n[Slide availability summary]")
    print(df_summary)

    # df_summary.to_csv("logs/slide_availability_summary.csv", index=False)
 
  
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

    # main(args)
    for fold_id in range(args.k_start, args.k_end + 1):
        print(f"Processing fold {fold_id}...")
        data_dir_map_configs=[
            'conch_patch_256x256_5x', 
            'conch_patch_256x256_10x'] 
        for data_dir_map_config in data_dir_map_configs:  
            check_data(fold_id, data_dir_map_config, args)
            print("---------------------------------------")

          
        

