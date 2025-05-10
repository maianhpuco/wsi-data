import openslide
import os 
# wsi = openslide.OpenSlide("/project/hnguyen2/mvu9/datasets/TGCA-datasets/KICH/bb078404-fd12-4f27-9d23-0fc412e76a52/TCGA-UW-A7GX-11Z-00-DX1.57C4FD28-5463-40C5-9E87-484F3326B395.svs")
# wsi = openslide.OpenSlide("~/datasets/TGCA-datasets/KICH/9ae213d2-e717-470d-93dc-f4e6f46d5425/TCGA-KL-8335-01Z-00-DX1.3F291515-CD98-4D8F-A255-A10A69139242.svs") 
path = os.path.expanduser("~/datasets/TGCA-datasets/KICH/9ae213d2-e717-470d-93dc-f4e6f46d5425/TCGA-KL-8335-01Z-00-DX1.3F291515-CD98-4D8F-A255-A10A69139242.svs")
wsi = openslide.OpenSlide(path) 
print(wsi.level_dimensions)

import pandas as pd
import os

# File paths
base_path = "/datasets/TGCA-metadata/KICH"
slides_path = os.path.join(base_path, "slides.xlsx")
uuids_path = os.path.join(base_path, "uuids.xlsx")

# Check existence
for file_path in [slides_path, uuids_path]:
    if not os.path.isfile(file_path):
        print(f"❌ File not found: {file_path}")
    else:
        print(f"✅ File exists: {file_path}")

# Load and inspect content
try:
    slides_df = pd.read_excel(slides_path)
    uuids_df = pd.read_excel(uuids_path)

    print(f"\nSlides.xlsx preview ({slides_df.shape[0]} rows):")
    print(slides_df.head())

    print(f"\nUuids.xlsx preview ({uuids_df.shape[0]} rows):")
    print(uuids_df.head())

    # Optional: check if filenames match
    slides_filenames = set(slides_df["Filename"])
    uuid_filenames = set(uuids_df["Filename"])

    unmatched = slides_filenames - uuid_filenames
    if unmatched:
        print(f"\n⚠️ {len(unmatched)} slide filenames not found in uuids.xlsx:")
        print(list(unmatched)[:5])
    else:
        print("\n✅ All slide filenames in slides.xlsx are found in uuids.xlsx.")

except Exception as e:
    print(f"🚨 Error reading Excel files: {e}")
