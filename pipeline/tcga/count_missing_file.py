import os
import pandas as pd

# Define PT file directories
dirs = {
    "KIRC": "/home/mvu9/processing_datasets/processing_tcga/kirc/features_fp/pt_files/",
    "KICH": "/home/mvu9/processing_datasets/processing_tcga/kich/features_fp/pt_files/",
    "KIRP": "/home/mvu9/processing_datasets/processing_tcga/kirp/features_fp/pt_files/"
}

# Define metadata Excel paths
metadata_files = {
    "KIRC": "/home/mvu9/datasets/TCGA-metadata/KIRC/uuids.xlsx",
    "KICH": "/home/mvu9/datasets/TCGA-metadata/KICH/uuids.xlsx",
    "KIRP": "/home/mvu9/datasets/TCGA-metadata/KIRP/uuids.xlsx"
}

# Count .pt file existence for each cancer subtype
for subtype in ["KIRC", "KICH", "KIRP"]:
    print(f"\nChecking subtype: {subtype}")
    df = pd.read_excel(metadata_files[subtype])
    df.columns = df.columns.str.lower()

    if 'filename' not in df.columns:
        raise ValueError(f"Expected 'filename' column not found in {metadata_files[subtype]}")

    df['filename'] = df['filename'].str.replace(".svs", "", regex=False)
    expected_files = df['filename'].unique()

    dir_path = dirs[subtype]
    if not os.path.exists(dir_path):
        print(f"⚠️  Directory does not exist: {dir_path}")
        continue

    available_files = {f.replace(".pt", "") for f in os.listdir(dir_path) if f.endswith(".pt")}

    found = [f for f in expected_files if f in available_files]
    missing = [f for f in expected_files if f not in available_files]

    print(f"  Total slides in metadata: {len(expected_files)}")
    print(f"  Found .pt files:          {len(found)}")
    print(f"  Missing .pt files:        {len(missing)}")

    if missing:
        print(f"  → Example missing files ({min(len(missing), 5)} shown):")
        for m in missing[:5]:
            print(f"    - {m}")
