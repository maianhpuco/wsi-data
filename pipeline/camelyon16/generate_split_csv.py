import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import yaml


def generate_split(df, split_folder, fold_number):
    # Step 1: Load the slide list
    df = pd.read_csv(args.slide_list)
    df.rename(columns={'slide_id': 'image'}, inplace=True)

    # Step 2: Load external test labels
    test_label_df = pd.read_csv('./camelyon16_csv_split_camil/splits_0.csv')
    test_label_map = dict(zip(test_label_df['test'], test_label_df['test_label']))

    # Step 3: Split based on image names
    train_val_df = df[df['image'].str.startswith(('normal', 'tumor'))].copy()
    test_df = df[df['image'].str.startswith('test')].copy()
    
    
    def clean_name_and_label(row):
        name = row['image'].replace('.tif', '')
        if name.startswith('normal'):
            label = 0
        elif name.startswith('tumor'):
            label = 1
        elif name.startswith('test'):
            label = test_label_map.get(name, None)  # get from external CSV
        else:
            label = None
        return pd.Series([name, label]) 
    
    
    train_val_df[['clean_name', 'label']] = train_val_df.apply(clean_name_and_label, axis=1)
    test_df[['clean_name', 'label']] = test_df.apply(clean_name_and_label, axis=1)

    # Step 2: Train-validation split
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=fold_number)

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Step 3: Create combined DataFrame
    min_len = min(len(train_df), len(val_df), len(test_df))
    split_df = pd.DataFrame({
        'train': train_df['clean_name'][:min_len],
        'train_label': train_df['label'][:min_len],
        'val': val_df['clean_name'][:min_len],
        'val_label': val_df['label'][:min_len],
        'test': test_df['clean_name'][:min_len],
        'test_label': test_df['label'][:min_len],
    })
    print("View the dataset")
    print(split_df.head(3))
    # Step 4: Save CSV
    split_df_path = os.path.join(split_folder, f'fold_{fold_number}.csv')
    os.makedirs(split_folder, exist_ok=True)
    split_df.to_csv(split_df_path, index=False)
    print(f"> Fold {fold_number} saved to: {split_df_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate multiple train/val/test splits for WSI dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    slide_list = config['paths']['slide_list']
    split_folder = config['paths']['split_folder']
    total_folds = int(config.get('fold_number', 5))  # default to 5 if not set

    # Read full dataset once
    df = pd.read_csv(slide_list)

    for fold_number in range(1, total_folds + 1):
        generate_split(df.copy(), split_folder, fold_number)
