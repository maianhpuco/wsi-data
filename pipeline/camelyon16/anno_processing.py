import os
import sys
import glob
import pandas as pd
import numpy as np
import argparse
import shutil
import h5py
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path) 
from utilities.camelyon16_annotation_processing import(
    extract_coordinates,
    check_xy_in_coordinates_fast
)

def read_h5_data(file_path, dataset_name=None):
    data = None
    with h5py.File(file_path, "r") as file:
        if dataset_name is not None:
            if dataset_name in file:
                dataset = file[dataset_name]
                data = dataset[()]
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found in the file.")
        else:
            datasets = {}
            def visitor(name, node):
                if isinstance(node, h5py.Dataset):
                    datasets[name] = node[()]
            file.visititems(visitor)
            if len(datasets) == 1:
                data = list(datasets.values())[0]
            else:
                data = datasets
    return data 
 
def main(args):
    _annotation_list = os.listdir(args.annotation_path)
    _h5_files = os.listdir(args.features_h5_path)

    annotation_list = [] 
    for anno_filename in _annotation_list:
        name = anno_filename.split(".")[0]
        if f"{name}.h5" in _h5_files: 
            annotation_list.append(anno_filename)

    total_file = len(annotation_list)
    print("Total files to process:", total_file)

    for idx, basename in enumerate(annotation_list):
        print(f"Processing {idx+1}/{total_file}: {basename}")
        name = basename.split(".")[0]
        h5_path = os.path.join(args.features_h5_path, f"{name}.h5")
        xml_path = os.path.join(args.annotation_path, f"{name}.xml")

        df_xml = extract_coordinates(xml_path)
        df_xml = pd.DataFrame(df_xml)

        h5_data = read_h5_data(h5_path)
        mask = check_xy_in_coordinates_fast(df_xml, h5_data["coordinates"])

        mask_save_path = os.path.join(args.ground_truth_path, f"{name}.npy")
        print("Saving mask to:", mask_save_path)
        np.save(mask_save_path, mask)
        break 
    mask_file = glob.glob(os.path.join(args.ground_truth_path, "*.npy")) 
    print("Total processed annotation files:", len(annotation_list))
    print("Generated mask files:", len(mask_file))

def reset_directory(path):
    if os.path.exists(path):
        print("Resetting directory:", path)
        shutil.rmtree(path)
    os.makedirs(path)  
 
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    # Load config YAML
    import yaml 
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.features_h5_path = config['paths']['ht_files']
    args.annotation_path = config['paths']['anno_xml_dir']
    args.ground_truth_path = config['paths']['mask_save_dir']
    os.makedirs(args.ground_truth_path, exist_ok=True)
    
    
    main(args)
