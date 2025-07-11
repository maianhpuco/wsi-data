import os
import pandas as pd
import torch
import h5py
from collections import defaultdict
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

        # Normalize single-scale dict input to nested format with '5x' key
        self.data_dir_map = {
            k: (v if isinstance(v, dict) else {'5x': v})
            for k, v in data_dir_map.items()
        }

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
        data_dir = self.data_dir_map[label.lower()]['5x']

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

def return_splits_custom(train_csv_path,
                          val_csv_path,
                          test_csv_path,
                          data_dir_map,
                          label_dict,
                          seed=1,
                          print_info=False,
                          use_h5=False,
                          args=None):

    global_kept_per_label = defaultdict(list)
    global_missing_log = defaultdict(list)

    def filter_df(df, name):
        print("------------------")
        kept, missing = [], []
        kept_per_label = defaultdict(list)
        missing_log = defaultdict(list)

        df = df.drop_duplicates(subset=["slide"])
        df['label'] = df['label'].str.lower()

        for _, row in df.iterrows():
            slide_id = row["slide"]
            label = row["label"]

            try:
                path = os.path.join(data_dir_map[label]['5x'] if isinstance(data_dir_map[label], dict) else data_dir_map[label],
                                     'h5_files' if use_h5 else 'pt_files',
                                     f"{slide_id}.h5" if use_h5 else f"{slide_id}.pt")
                if os.path.exists(path):
                    kept.append(row)
                    kept_per_label[label].append(slide_id)
                    global_kept_per_label[label].append(slide_id)
                else:
                    missing.append(row)
                    missing_log[label].append(slide_id)
                    global_missing_log[label].append(slide_id)
            except Exception as e:
                print(f"[WARN] {slide_id} → error: {e}")
                missing.append(row)
                missing_log[label].append(slide_id)
                global_missing_log[label].append(slide_id)

        df_kept = pd.DataFrame(kept).drop_duplicates(subset=["slide"])
        df_missing = pd.DataFrame(missing).drop_duplicates(subset=["slide"])

        os.makedirs("logs", exist_ok=True)
        if missing_log:
            pd.DataFrame([(k, v) for k, lst in missing_log.items() for v in lst],
                         columns=["label", "slide"]).to_csv(f"logs/missing_slides_{name}.csv", index=False)
            print(f"[INFO] Saved missing slides → logs/missing_slides_{name}.csv")

        print(f"[INFO] {name.upper()}: Kept {len(df_kept)} / {len(df)}")

        if args is not None and getattr(args, "dataset_name", "") == "tcga_renal":
            labels = ['kich', 'kirc', 'kirp']
        elif args is not None and getattr(args, "dataset_name", "") == "tcga_lung":
            labels = ['luad', 'lusc']
        else:
            labels = df["label"].dropna().unique().tolist()

        for label in labels:
            label = label.lower()
            count_miss = len(missing_log[label])
            available = len(kept_per_label[label])
            print(f"[SUMMARY - {name.upper()} | {label.upper()}] -- AVAILABLE: {available}  MISSING: {count_miss}")

        return df_kept, df_missing

    def create_dataset(df):
        return Generic_MIL_Dataset(
            data_dir_map=data_dir_map,
            patient_ids=df["patient_id"].dropna().tolist(),
            slides=df["slide"].dropna().tolist(),
            labels=df["label"].dropna().tolist(),
            label_dict=label_dict,
            seed=seed,
            print_info=print_info,
            use_h5=use_h5
        )

    df_train, df_train_missing = filter_df(pd.read_csv(train_csv_path), "train")
    df_val, df_val_missing = filter_df(pd.read_csv(val_csv_path), "val")
    df_test, df_test_missing = filter_df(pd.read_csv(test_csv_path), "test")

    # Save combined summary
    summary_rows = []
    labels_all = set(list(global_kept_per_label.keys()) + list(global_missing_log.keys()))
    for label in sorted(labels_all):
        total_available = len(global_kept_per_label[label])
        total_missing = len(global_missing_log[label])
        summary_rows.append((label, total_available, total_missing))
        print(f"[SUMMARY - ALL | {label.upper()}] -- AVAILABLE: {total_available}  MISSING: {total_missing}")

    df_summary = pd.DataFrame(summary_rows, columns=["label", "available", "missing"])
    df_summary.to_csv("logs/missing_summary_all.csv", index=False)
    print("[INFO] Saved combined summary to logs/missing_summary_all.csv")

    return create_dataset(df_train), create_dataset(df_val), create_dataset(df_test)
