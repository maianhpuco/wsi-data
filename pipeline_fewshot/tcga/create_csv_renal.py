import pandas as pd
import os

# Define input Excel files for each subtype
subtypes = {
    "KIRC": "/home/mvu9/datasets/TCGA-metadata/KIRC/uuids.xlsx",
    "KIRP": "/home/mvu9/datasets/TCGA-metadata/KIRP/uuids.xlsx",
    "KICH": "/home/mvu9/datasets/TCGA-metadata/KICH/uuids.xlsx"
}

# Mapping to standard names
subtype_label_map = {
    "KIRC": "ccRCC",
    "KIRP": "pRCC",
    "KICH": "chRCC"
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
final_df.to_csv("/home/mvu9/datasets/TCGA-metadata/TCGA_RCC_subtyping.csv", index=False)
