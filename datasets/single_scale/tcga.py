import os
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset

class Generic_MIL_Dataset(Dataset):
    def __init__(self,
                 data_dir_map,
                 patient_ids,
                 slides,
                 labels,
                 label_dict,
                 seed=1,
                 print_info=False,
                 use_h5=False, 
                 ignore=[],
                 **kwargs):

        self.data_dir_map = data_dir_map  # expects nested dict for multi-scale
        self.label_dict = label_dict
        self.ignore = ignore
        self.seed = seed
        self.use_h5 = use_h5
        self.kwargs = kwargs

        self.slide_data = pd.DataFrame({
            'patient_id': patient_ids,
            'slide_id': slides,
            'label': labels
        })

        if print_info:
            print("Loaded {} samples from {}".format(len(self.slide_data), data_dir_map))

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'].iloc[idx]
        label = self.slide_data['label'].iloc[idx]

        data_dir = self.data_dir_map[label.lower()]['5x']  # Default scale for __getitem__

        if not self.use_h5:
            full_path = os.path.join(data_dir, 'pt_files', f"{slide_id}.pt")
            features = torch.load(full_path, weights_only=True)
            return features, self.label_dict[label], label
        else:
            full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
            with h5py.File(full_path, 'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
            features = torch.from_numpy(features)
            return features, self.label_dict[label], coords

    def get_features_by_slide_id(self, slide_id):
        row = self.slide_data[self.slide_data['slide_id'] == slide_id]
        if row.empty:
            raise ValueError(f"Slide ID {slide_id} not found in dataset.")

        label = row.iloc[0]['label']
        data_dir = self.data_dir_map[label.lower()]['5x']  # Default to 5x

        if not self.use_h5:
            full_path = os.path.join(data_dir, 'pt_files', f"{slide_id}.pt")
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Feature file not found: {full_path}")
            feats = torch.load(full_path)
        else:
            full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"H5 file not found: {full_path}")
            with h5py.File(full_path, 'r') as hdf5_file:
                feats = torch.from_numpy(hdf5_file['features'][:])

        if feats.ndim == 1:
            feats = feats.unsqueeze(0)

        return feats

    def count_missing_files_multiscale(self, scales=['5x', '10x']):
        missing = {}
        for _, row in self.slide_data.iterrows():
            slide_id = row['slide_id']
            label = row['label'].lower()
            missing_scales = []

            for scale in scales:
                if label not in self.data_dir_map or scale not in self.data_dir_map[label]:
                    print(f"Missing directory mapping for label {label} at scale {scale}")
                    continue

                data_dir = self.data_dir_map[label][scale]
                folder = 'h5_files' if self.use_h5 else 'pt_files'
                filename = f"{slide_id}.h5" if self.use_h5 else f"{slide_id}.pt"
                full_path = os.path.join(data_dir, folder, filename)

                if not os.path.exists(full_path):
                    missing_scales.append(scale)

            if missing_scales:
                missing[slide_id] = missing_scales

        print(f"Total slides with missing files: {len(missing)}")
        return missing


def return_splits_custom(train_csv_path, 
                          val_csv_path, 
                          test_csv_path, 
                          data_dir_map, 
                          label_dict, 
                          seed=1, 
                          print_info=False, 
                          use_h5=False):

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)

    def create_dataset(df):
        patient_ids = df["patient_id"].dropna().tolist()
        slides = df["slide"].dropna().tolist()
        labels = df["label"].dropna().tolist()
        return Generic_MIL_Dataset(
            data_dir_map=data_dir_map,
            patient_ids=patient_ids,
            slides=slides,
            labels=labels,
            label_dict=label_dict,
            shuffle=False,
            seed=seed,
            print_info=print_info, 
            use_h5=use_h5
        )

    train_dataset = create_dataset(df_train)
    val_dataset = create_dataset(df_val)
    test_dataset = create_dataset(df_test)

    return train_dataset, val_dataset, test_dataset
