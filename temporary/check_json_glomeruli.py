import json
import os
import glob
from pathlib import Path

# Set your annotation root directory
ANN_ROOT = "/home/mvu9/datasets/glomeruli/annotations"  # update if needed

# Get the first JSON file
json_files = sorted(glob.glob(os.path.join(ANN_ROOT, "*.json")))
if not json_files:
    print("No JSON files found!")
else:
    json_path = json_files[0]
    print(f"Reading file: {json_path}")

    # Load JSON and print top-level keys
    with open(json_path, 'r') as f:
        data = json.load(f)
        print("Top-level keys in JSON:")
        for key in data.keys():
            print(f" - {key}")
