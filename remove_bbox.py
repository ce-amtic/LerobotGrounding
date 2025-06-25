import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from batched_grounding import load_parquet

def drop_bbox_columns_from_parquet(parquet_path: str):
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[ERROR] Failed to read {parquet_path}: {e}")
        return False

    changed = False
    for col in ['bbox']: #, 'bbox_index']:
        if col in df.columns:
            df = df.drop(columns=[col])
            changed = True

    if changed:
        try:
            df.to_parquet(parquet_path)
            print(f"[OK] Removed columns and saved: {parquet_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save {parquet_path}: {e}")
            return False
    else:
        print(f"[SKIP] No bbox columns in: {parquet_path}")
    
    return True

def remove_bboxes_from_dataset(dataset_path: str):
    print(f"Removing bbox columns in dataset: {dataset_path}")
    all_parquet_files = load_parquet(dataset_path)
    for parquet_path in tqdm(all_parquet_files, desc="Dropping bbox cols", unit="file"):
        drop_bbox_columns_from_parquet(parquet_path)

if __name__ == "__main__":
    all_datasets = [
        # '/pdata/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds',
        '/pdata/oxe_lerobot/berkeley_fanuc_manipulation'
    ]

    for i, dataset_path in enumerate(all_datasets):
        print(f"\n[{i+1}/{len(all_datasets)}] Processing {dataset_path}")
        remove_bboxes_from_dataset(dataset_path)
