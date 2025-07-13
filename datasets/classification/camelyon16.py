import os
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset

class Generic_MIL_Dataset(Dataset):
    def __init__(self,
                 data_dir,
                 csv_path=None,
                 label_dict=None,
                 case_ids=None,
                 labels=None,
                 seed=1,
                 print_info=False,
                 use_h5=False, 
                 ignore=[],
                 **kwargs):
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.ignore = ignore
        self.seed = seed
        self.use_h5 = use_h5
        self.kwargs = kwargs

        # Load slides and labels
        if case_ids is not None and labels is not None:
            self.slide_data = pd.DataFrame({
                'slide_id': case_ids,
                'label': labels
            })
        elif csv_path is not None:
            df = pd.read_csv(csv_path)
            self.slide_data = df[['slide_id', 'label']].copy()
        else:
            raise ValueError("Must provide either (csv_path) or (case_ids and labels).")

        # Filter out ignored slides
        self.slide_data = self.slide_data[~self.slide_data['slide_id'].isin(ignore)]

        if print_info:
            print("Loaded {} samples from {}".format(len(self.slide_data), data_dir))

    def load_from_h5(self, toggle):
        """Toggle between .pt and .h5 file loading."""
        self.use_h5 = toggle

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'].iloc[idx]
        label = self.slide_data['label'].iloc[idx]

        # Determine data directory
        if isinstance(self.data_dir, dict):
            # Assume single source for simplicity; extend if needed
            source = self.kwargs.get('source', list(self.data_dir.keys())[0])
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not data_dir:
            return slide_id, label

        if not self.use_h5:
            full_path = os.path.join(data_dir, 'pt_files', f"{slide_id}.pt")
            features = torch.load(full_path, weights_only=True)
            return features, label
        else:
            full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
            with h5py.File(full_path, 'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
            features = torch.from_numpy(features)
            return features, label, coords
        
    def get_features_by_slide_id(self, slide_id):
        """
        Load features for a specific slide ID.

        Args:
            slide_id (str): The slide ID whose features to load.

        Returns:
            torch.Tensor: Feature tensor for the slide.
        """
        if isinstance(self.data_dir, dict):
            source = self.kwargs.get('source', list(self.data_dir.keys())[0])
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not data_dir:
            raise ValueError("No data directory provided.")

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

        if len(feats.shape) == 1:
            feats = feats.unsqueeze(0)
        return feats
 
def return_splits_custom(csv_path, data_dir, label_dict, seed=1, print_info=False, use_h5=False):
    """Create train, val, and test datasets from a custom split CSV."""
    df = pd.read_csv(csv_path)

    def create_dataset(col_case, col_label):
        case_ids = df[col_case].dropna().tolist()
        labels = df[col_label].dropna().astype(int).tolist()
        return Generic_MIL_Dataset(
            data_dir=data_dir,
            case_ids=case_ids,
            labels=labels,
            label_dict=label_dict,
            shuffle=False,
            seed=seed,
            print_info=print_info, 
            use_h5=use_h5
        )

    train_dataset = create_dataset('train', 'train_label')
    val_dataset = create_dataset('val', 'val_label')
    test_dataset = create_dataset('test', 'test_label')

    return train_dataset, val_dataset, test_dataset
