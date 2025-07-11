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
                 use_h5=True,
                 ignore=[],
                 **kwargs):
        self.data_dir_map = data_dir_map
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
        # print("-------")
        # print(self.slide_data)
        if print_info:
            print(f"Loaded {len(self.slide_data)} slides.")

    def __len__(self):
        return len(self.slide_data)

    def _resolve_subtype_path(self, slide_id, path_dict):
        for key in path_dict:
            if key.lower() in slide_id.lower():
                return path_dict[key]
        raise ValueError(f"Cannot match slide_id '{slide_id}' to any subtype in {list(path_dict.keys())}")
    
    def __getitem__(self, idx):
        row = self.slide_data.iloc[idx]
        slide_id = row['slide_id']
        label_str = row['label']
        label = self.label_dict[label_str]

        folder_data = self.data_dir_map[label_str.lower()]
        
        h5_path = os.path.join(folder_data, f"{slide_id}.h5")
        with h5py.File(h5_path, 'r') as f_s:
            features = torch.from_numpy(f_s['features'][:])
            coords   = torch.from_numpy(f_s['coords'][:])

        
        return features, coords, label

# def return_splits_custom(train_csv_path,
#                           val_csv_path,
#                           test_csv_path,
#                           data_dir_map,
#                           label_dict,
#                           seed=1,
#                           print_info=False,
#                           use_h5=False,
#                           args=None):

#     def filter_df(df, name):
#         print("------------------")
#         kept, missing = [], []
#         kept_per_label = defaultdict(list)
#         missing_log = defaultdict(list)

#         df = df.drop_duplicates(subset=["slide"])
#         print(df.head(5))
#         print(df.columns)
#         for _, row in df.iterrows():
#             slide_id = row["slide"]
#             label = row["label"].lower()

#             try:
#                 path = os.path.join(data_dir_map[label], f"{slide_id}.h5")
#                 if os.path.exists(path):
#                     kept.append(row)
#                     kept_per_label[label].append(slide_id)
#                 else:
#                     missing.append(row)
#                     missing_log[label].append(slide_id)
#             except Exception as e:
#                 print(f"[WARN] {slide_id} → error: {e}")
#                 missing.append(row)
#                 missing_log[label].append(slide_id)

#         df_kept = pd.DataFrame(kept).drop_duplicates(subset=["slide"])

#         os.makedirs("logs", exist_ok=True)
#         if missing_log:
#             pd.DataFrame([(k, v) for k, lst in missing_log.items() for v in lst],
#                          columns=["label", "slide"]).to_csv(f"logs/missing_slides_{name}.csv", index=False)
#             print(f"[INFO] Saved missing slides → logs/missing_slides_{name}.csv")

#         print(f"[INFO] {name.upper()}: Kept {len(df_kept)} / {len(df)}")

#         if args is not None and getattr(args, "dataset_name", "") == "tcga_renal":
#             labels = ['kich', 'kirc', 'kirp']
#         elif args is not None and getattr(args, "dataset_name", "") == "tcga_lung":
#             labels = ['luad', 'lusc']
#         else:
#             labels = df["label"].dropna().unique().tolist()
#         return df_kept

#     def create_dataset(df):
#         return Generic_MIL_Dataset(
#             data_dir_map=data_dir_map,
#             patient_ids=df["patient_id"].dropna().tolist(),
#             slides=df["slide"].dropna().tolist(),
#             labels=df["label"].dropna().tolist(),
#             label_dict=label_dict,
#             seed=seed,
#             print_info=print_info,
#             use_h5=use_h5
#         )

#     df_train = filter_df(pd.read_csv(train_csv_path), "train")
#     df_val = filter_df(pd.read_csv(val_csv_path), "val")
#     df_test = filter_df(pd.read_csv(test_csv_path), "test")

#     return create_dataset(df_train), create_dataset(df_val), create_dataset(df_test)
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
                path = os.path.join(data_dir_map[label])
                print(path) 
                # path = os.path.join(data_dir_map[label]['5x'] if isinstance(data_dir_map[label], dict) else data_dir_map[label],
                #                     'h5_files' if use_h5 else 'pt_files',
                #                     f"{slide_id}.h5" if use_h5 else f"{slide_id}.pt")
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
    # Return datasets and DataFrames
    # return {
    #     'train_dataset': create_dataset(df_train),
    #     'val_dataset': create_dataset(df_val),
    #     'test_dataset': create_dataset(df_test),
    #     'train_kept': df_train,
    #     'train_missing': df_train_missing,
    #     'val_kept': df_val,
    #     'val_missing': df_val_missing,
    #     'test_kept': df_test,
    #     'test_missing': df_test_missing
    # }