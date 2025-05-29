import os
import glob
import shutil

# Define source and destination directories
base_src_dir = "/home/mvu9/datasets/kidney_pathology_image"
base_dst_dir = "/home/mvu9/processing_datasets/kidney_pathology_image/reorganized_rawdata"

# Define splits
splits = ["train", "test", "validation"]

# Source patterns
slide_dir = {
    "train": f"{base_src_dir}/train/Task2_WSI_level/*/*_wsi.tiff",
    "test": f"{base_src_dir}/test/Task2_WSI_level/*/*_wsi.tiff",
    "validation": f"{base_src_dir}/validation/Task2_WSI_level/*/*_wsi.tiff"
}

anno_xml_dir = {
    "train": f"{base_src_dir}/train/Task2_WSI_level/*/*_mask.tiff",
    "test": f"{base_src_dir}/test/Task2_WSI_level/*/*_mask.tiff",
    "validation": f"{base_src_dir}/validation/Task2_WSI_level/*/*_mask.tiff"
}

# Destination directories
image_dst_dir = {
    "train": f"{base_dst_dir}/images/train",
    "test": f"{base_dst_dir}/images/test",
    "validation": f"{base_dst_dir}/images/validation"
}

mask_dst_dir = {
    "train": f"{base_dst_dir}/annotations/train",
    "test": f"{base_dst_dir}/annotations/test",
    "validation": f"{base_dst_dir}/annotations/validation"
}

# Function to create directory if it doesn't exist
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Function to move files
def move_files(src_pattern, dst_dir, file_type):
    # Create destination directory
    create_dir(dst_dir)
    
    # Get all files matching the pattern
    files = glob.glob(src_pattern, recursive=True)
    
    if not files:
        print(f"No {file_type} files found for pattern: {src_pattern}")
        return
    
    # Move each file to the destination
    for src_file in files:
        file_name = os.path.basename(src_file)
        dst_file = os.path.join(dst_dir, file_name)
        
        # Move the file
        shutil.move(src_file, dst_file)
        print(f"Moved {file_type} file: {src_file} -> {dst_file}")

# Process each split
for split in splits:
    print(f"\nProcessing {split} split...")
    
    # Move image files
    move_files(slide_dir[split], image_dst_dir[split], "image")
    
    # Move mask files
    move_files(anno_xml_dir[split], mask_dst_dir[split], "mask")

print("\nFile moving completed.")