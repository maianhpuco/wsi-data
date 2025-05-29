import pandas as pd
import os

# Define input Excel files for each subtype
subtypes = {
    "KIRC": "/home/mvu9/datasets/TCGA-metadata/LUAD/uuids.xlsx",
    "KIRP": "/home/mvu9/datasets/TCGA-metadata/LUSD/uuids.xlsx",
}

# Mapping to standard names
subtype_label_map = {
    "LUAD": "LUAD",
    "LUSC": "LUSC",
}

# Prepare rows
all_rows = []

for subtype, filepath in subtypes.items():
    df = pd.read_excel(filepath)
    label = subtype_label_map[subtype]

    for idx, row in df.iterrows():
        case_id = f"patient_{idx}"
        slide_id = row["Filename"].replace(".svs", "")
        all_rows.append((case_id, slide_id, label))

# Create final dataframe and save
final_df = pd.DataFrame(all_rows, columns=["case_id", "slide_id", "label"])
final_df.to_csv("/home/mvu9/datasets/TCGA-metadata/TCGA_LUNG_subtyping.csv", index=False)
