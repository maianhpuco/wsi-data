import os
import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split

def count_labels(df):
    counts = df['label'].value_counts().to_dict()
    return ', '.join([f'{label}: {counts.get(label, 0)}' for label in ['KICH', 'KIRP', 'KIRC']])

def generate_tcga_splits(config):
    kich_path = config['paths']['metadata']['kich']
    kirp_path = config['paths']['metadata']['kirp']
    kirc_path = config['paths']['metadata']['kirc']
    pt_dirs = config['paths']['pt_files_dir']
    output_dir = config['paths']['split_folder']
    num_folds = int(config.get('fold_number', 5))

    all_pt_files = set()
    for cancer_type in ['kich', 'kirp', 'kirc']:
        pt_dir = pt_dirs[cancer_type]
        pt_files = [os.path.splitext(f)[0] for f in os.listdir(pt_dir) if f.endswith('.pt')]
        all_pt_files.update(pt_files)
    print(f"Found {len(all_pt_files)} total .pt files across KICH, KIRP, and KIRC")

    kich_df = pd.read_excel(kich_path)
    kich_df['label'] = 'KICH'
    kirp_df = pd.read_excel(kirp_path)
    kirp_df['label'] = 'KIRP'
    kirc_df = pd.read_excel(kirc_path)
    kirc_df['label'] = 'KIRC'

    df = pd.concat([kich_df, kirp_df, kirc_df], ignore_index=True)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={'uuid': 'patient_id', 'filename': 'slide'})
    df['slide'] = df['slide'].str.replace('.svs', '', regex=False)

    df = df[df['slide'].isin(all_pt_files)]
    print(f"Filtered to {len(df)} slides with existing .pt files")

    unique_patients = df[['patient_id', 'label']].drop_duplicates()
    patient_ids = unique_patients['patient_id'].values

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    folds_patients = []

    for _, (train_index, test_index) in enumerate(kf.split(patient_ids)):
        train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=42)
        folds_patients.append({
            'train': patient_ids[train_index],
            'validation': patient_ids[val_index],
            'test': patient_ids[test_index]
        })

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_folds):
        fold_id = i + 1
        print(f"\nFold {fold_id}")

        fold_dir = os.path.join(output_dir, f'fold_{fold_id}')
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df[df['patient_id'].isin(folds_patients[i]['train'])]
        val_df = df[df['patient_id'].isin(folds_patients[i]['validation'])]
        test_df = df[df['patient_id'].isin(folds_patients[i]['test'])]

        print(train_df.head(3))
        print(f"Train: {len(train_df)} samples -> {count_labels(train_df)}")
        print(f"Val:   {len(val_df)} samples -> {count_labels(val_df)}")
        print(f"Test:  {len(test_df)} samples -> {count_labels(test_df)}")
        print(f"Total: {len(train_df) + len(val_df) + len(test_df)} / {len(df)} complete")

        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

        print(f"Saved to: {fold_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate TCGA patient-level WSI cross-validation splits')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    generate_tcga_splits(config)
