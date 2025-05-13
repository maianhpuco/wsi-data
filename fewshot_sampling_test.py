import os
import shutil
import random

# Directories
source_root = '/home/mvu9/processing_datasets/processing_tgca'
fewshot_root = '/home/mvu9/processing_datasets/fewshot_tgca_4_shots'
validation_root = '/home/mvu9/processing_datasets/validation'
test_root = '/home/mvu9/processing_datasets/test_tgca'
os.makedirs(test_root, exist_ok=True)

classes = ['kich', 'kirc', 'kirp']
num_samples = 3

for cls in classes:
    src_patch_dir = os.path.join(source_root, cls, 'patches_png')
    fewshot_dir = os.path.join(fewshot_root, cls, 'patches_png')
    valid_dir = os.path.join(validation_root, cls, 'patches_png')
    test_dir = os.path.join(test_root, cls, 'patches_png')
    os.makedirs(test_dir, exist_ok=True)

    # All folders in the original source
    all_folders = {
        f for f in os.listdir(src_patch_dir)
        if os.path.isdir(os.path.join(src_patch_dir, f))
    }

    # Folders used in fewshot and validation
    used_folders = set()
    for d in [fewshot_dir, valid_dir]:
        if os.path.exists(d):
            used_folders.update(
                f for f in os.listdir(d)
                if os.path.isdir(os.path.join(d, f))
            )

    # Remaining folders for test
    available_folders = list(all_folders - used_folders)

    # Randomly sample up to num_samples
    sampled_folders = random.sample(available_folders, min(num_samples, len(available_folders)))

    for folder in sampled_folders:
        src_path = os.path.join(src_patch_dir, folder)
        dest_path = os.path.join(test_dir, folder)
        shutil.copytree(src_path, dest_path)
        print(f"Copied {src_path} -> {dest_path}")
