import os
import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from collections import Counter


def count_labels(df):
    counts = df['label'].value_counts().to_dict()
    return ', '.join([f'{label}: {counts.get(label, 0)}' for label in ['KICH', 'KIRP', 'KIRC']])


def generate_tcga_splits(config):
    # Read config paths
    kich_path = config['paths']['metadata']['kich']
    kirp_path = config['paths']['metadata']['kirp']
    kirc_path = config['paths']['metadata']['kirc']
    output_dir = config['paths']['split_folder']
    num_folds = int(config.get('fold_number', 5))

    # Load Excel files and assign labels
    kich_df = pd.read_excel(kich_path)
    kich_df['label'] = 'KICH'

    kirp_df = pd.read_excel(kirp_path)
    kirp_df['label'] = 'KIRP'

    kirc_df = pd.read_excel(kirc_path)
    kirc_df['label'] = 'KIRC'

    # Combine and rename
    df = pd.concat([kich_df, kirp_df, kirc_df], ignore_index=True)
    df.columns = df.columns.str.strip().str.lower()  # normalize column names
    df = df.rename(columns={'uuid': 'patient_id', 'filename': 'slide'})
    df['slide'] = df['slide'].str.replace('.svs', '', regex=False)

    # Unique patients
    unique_patients = df[['patient_id', 'label']].drop_duplicates()
    patient_ids = unique_patients['patient_id'].values

    # Set up KFold on patients
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    folds_patients = []

    for i, (train_index, test_index) in enumerate(kf.split(patient_ids)):
        train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=42)
        folds_patients.append({
            'train': patient_ids[train_index],
            'validation': patient_ids[val_index],
            'test': patient_ids[test_index]
        })

    os.makedirs(output_dir, exist_ok=True)

    # Save each fold
    for i in range(num_folds):
        fold_id = i + 1
        print(f"\n Fold {fold_id}")

        fold_dir = os.path.join(output_dir, f'fold_{fold_id}')
        os.makedirs(fold_dir, exist_ok=True)

        # Get patient-level splits
        train_patients = folds_patients[i]['train']
        val_patients = folds_patients[i]['validation']
        test_patients = folds_patients[i]['test']

        # Map back to slides
        train_df = df[df['patient_id'].isin(train_patients)]
        val_df = df[df['patient_id'].isin(val_patients)]
        test_df = df[df['patient_id'].isin(test_patients)]

        # Show class balance
        print(train_df.head(3))
        print(f"Train: {len(train_df)} samples → {count_labels(train_df, 'label')}")
        print(f"Val:   {len(val_df)} samples → {count_labels(val_df, 'label')}")
        print(f"Test:  {len(test_df)} samples → {count_labels(test_df, 'label')}")
        total = len(train_df) + len(val_df) + len(test_df)
        print(f"Total: {total} / {len(df)} complete")

        # Save
        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

        print(f" Saved to: {fold_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate TCGA patient-level WSI cross-validation splits')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    generate_tcga_splits(config)
