import os
import argparse
import yaml
from tqdm import tqdm
import h5py
import openslide
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import glob 
import random 
import shutil 

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

 
def main(args, config): 
    
    dest_parent_folder = '/project/hnguyen2/mvu9/sample_data' 
    NUM_FILES = 5 
    # data_names = ['kich', 'kirp', 'kirc', 'luad', 'luad']
    # for name in data_names: 
    dataset_name = config['dataset_name']
    print(dataset_name)
    file_paths = glob.glob(f"/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/{dataset_name}/png_patches/patch_256x256_10x/*")
    print("Total file", len(file_paths))
    num_file = len(file_paths)
    sample_indexes = random.sample(range(0, num_file), 3) 
    # sample_indexes = [random.randint(0, NUM_FILES) for i in range(0, NUM_FILES)] 
    print(sample_indexes) 
    dest_folder = os.path.join(dest_parent_folder, config['dataset_name']) 
    os.makedirs(os.path.join(dest_folder), exist_ok=True)
    
    for idx in sample_indexes: 
        
        src_file = file_paths[idx] 
        basename = os.path.basename(src_file) 
        dest_file = os.path.join(dest_folder, basename)
        print("src: ", src_file)
        print("dest: ", dest_file)
        if os.path.exists(dest_file):
            shutil.rmtree(dest_file)
        shutil.copytree(src_file, dest_file)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    main(args, config)
 

