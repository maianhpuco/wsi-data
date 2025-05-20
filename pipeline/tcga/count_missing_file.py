import os
import pandas as pd

# Define PT file directories
pt_dirs = {
    "KIRC": "/home/mvu9/processing_datasets/processing_tcga/kirc/features_fp/pt_files/",
    "KICH": "/home/mvu9/processing_datasets/processing_tcga/kich/features_fp/pt_files/",
    "KIRP": "/home/mvu9/processing_datasets/processing_tcga/kirp/features_fp/pt_files/"
}

# Define H5 file directories
h5_dirs = {
    "KIRC": "/home/mvu9/processing_datasets/processing_tcga/kirc/patches_h5/",
    "KICH": "/home/mvu9/processing_datasets/processing_tcga/kich/patches_h5/",
    "KIRP": "/home/mvu9/processing_datasets/processing_tcga/kirp/patches_h5/"
}

# Define metadata Excel paths
metadata_files = {
    "KIRC": "/home/mvu9/datasets/TCGA-metadata/KIRC/uuids.xlsx",
    "KICH": "/home/mvu9/datasets/TCGA-metadata/KICH/uuids.xlsx",
    "KIRP": "/home/mvu9/datasets/TCGA-metadata/KIRP/uuids.xlsx"
}

# Check both .pt and .h5 files
for subtype in ["KICH", "KIRP"]:
    print(f"\n🧪 Checking subtype: {subtype}")
    df = pd.read_excel(metadata_files[subtype])
    df.columns = df.columns.str.lower()

    if 'filename' not in df.columns:
        raise ValueError(f"Expected 'filename' column not found in {metadata_files[subtype]}")

    df['filename'] = df['filename'].str.replace(".svs", "", regex=False)
    expected_files = df['filename'].unique()

    # ----- Check .pt files -----
    pt_path = pt_dirs[subtype]
    if not os.path.exists(pt_path):
        print(f"⚠️  .pt directory does not exist: {pt_path}")
    else:
        available_pt = {f.replace(".pt", "") for f in os.listdir(pt_path) if f.endswith(".pt")}
        missing_pt = [f for f in expected_files if f not in available_pt]

        print(f"📦 .pt Files")
        print(f"  Found:    {len(expected_files) - len(missing_pt)}")
        print(f"  Missing:  {len(missing_pt)}")
        if missing_pt:
            print(f"  → Example missing .pt files ({min(len(missing_pt), 5)} shown):")
            for m in missing_pt[:5]:
                print(f"    - {m}")

    # ----- Check .h5 files -----
    h5_path = h5_dirs[subtype]
    if not os.path.exists(h5_path):
        print(f"⚠️  .h5 directory does not exist: {h5_path}")
    else:
        missing_h5 = [f for f in expected_files if not os.path.isfile(os.path.join(h5_path, f"{f}.h5"))]

        print(f"🧬 .h5 Files")
        print(f"  Found:    {len(expected_files) - len(missing_h5)}")
        print(f"  Missing:  {len(missing_h5)}")
        if missing_h5:
            print(f"  → Example missing .h5 files ({min(len(missing_h5), 5)} shown):")
            for m in missing_h5[:5]:
                print(f"    - {m}")
