import os
import pandas as pd

# Define paths
dirs = {
    "KIRC": "/home/mvu9/processing_datasets/processing_tcga/kirc/features_fp/pt_files/",
    "KICH": "/home/mvu9/processing_datasets/processing_tcga/kich/features_fp/pt_files/",
    "KIRP": "/home/mvu9/processing_datasets/processing_tcga/kirp/features_fp/pt_files/"
}

metadata_files = {
    "KIRC": "/home/mvu9/datasets/TCGA-metadata/KIRC/uuids.xlsx",
    "KICH": "/home/mvu9/datasets/TCGA-metadata/KICH/uuids.xlsx",
    "KIRP": "/home/mvu9/datasets/TCGA-metadata/KIRP/uuids.xlsx"
}

# Count existing .pt files for each subtype
for subtype in ["KIRC", "KICH", "KIRP"]:
    df = pd.read_excel(metadata_files[subtype])
    
    print(df.head(3))
    df.columns = df.columns.str.lower()
    if 'file_name' not in df.columns:
        raise ValueError(f"'file_name' column not found in {metadata_files[subtype]}")

    df['file_name'] = df['file_name'].str.replace(".svs", "", regex=False)
    expected_files = df['file_name'].unique()
    
    dir_path = dirs[subtype]
    available_files = set(f.replace(".pt", "") for f in os.listdir(dir_path) if f.endswith(".pt"))

    found = [f for f in expected_files if f in available_files]
    missing = [f for f in expected_files if f not in available_files]

    print(f"\n{subtype}:")
    print(f"  Expected .pt files: {len(expected_files)}")
    print(f"  Found .pt files:    {len(found)}")
    print(f"  Missing .pt files:  {len(missing)}")
