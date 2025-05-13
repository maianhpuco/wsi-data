import os
import shutil
import random

# Root directory containing kich, kirc, kirp
source_root = '/home/mvu9/processing_datasets/processing_tgca'

# Destination directory
# dest_root = '/home/mvu9/processing_datasets/fewshot_tgca_16_shots'
dest_root = '/home/mvu9/processing_datasets/validation'
if os.path.exists(dest_root):
    # Remove the directory if it exists
    shutil.rmtree(dest_root) 
os.makedirs(dest_root, exist_ok=True)

classes = ['kich', 'kirc', 'kirp']
num_samples = 2 

for cls in classes:
    src_patch_dir = os.path.join(source_root, cls, 'patches_png')
    dest_patch_dir = os.path.join(dest_root, cls, 'patches_png')
    os.makedirs(dest_patch_dir, exist_ok=True)

    # Get list of patient subfolders
    patient_folders = [
        f for f in os.listdir(src_patch_dir)
        if os.path.isdir(os.path.join(src_patch_dir, f))
    ]

    # Randomly sample 4 folders
    sampled = random.sample(patient_folders, min(num_samples, len(patient_folders)))

    for folder in sampled:
        src_path = os.path.join(src_patch_dir, folder)
        dest_path = os.path.join(dest_patch_dir, folder)
        shutil.copytree(src_path, dest_path)

        print(f"Copied {src_path} -> {dest_path}")