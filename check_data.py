import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from utils.file_utils import save_pkl
from utils.core_utils_desc import train  # Make sure this expects model as the first arg

sys.path.append(os.path.join("src"))  
sys.path.append("src")
os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/cache_folder/.cache/huggingface'
from explainer_ver2 import Ver2f
import ml_collections

# === PATH SETUP ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'src')))

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def main(args):
    # import json

    # with open(args.text_prompts_path, "r") as f:
    #     args.text_prompts = json.load(f)
    # print(args.text_prompts)    
    
    # seed_torch(args.seed)

    all_test_auc, all_val_auc, all_test_acc, all_val_acc, all_test_f1, folds = [], [], [], [], [], []

    for i in range(args.k_start, args.k_end + 1):
        datasets = prepare_dataset(args, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_dir_map_config', type=str, default=None, help='Path to the data directory mapping file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    print("################# SETTINGS ###################")
    for k, v in vars(args).items():
        if k != 'paths':
            print(f"{k}: {v}")
    print("##############################################")

    main(args)

