import json, sys, logging, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from check_bbox import load_parquet

# def load_parquet(dataset_path, max_files):
#     parquet_paths = []

#     if type(dataset_path) is str:
#         dataset_path = Path(dataset_path)
#     if not dataset_path.exists():
#         logging.error(f"Dataset in {dataset_path} does not exist.")
#         sys.exit(1)
    
#     data_root = dataset_path / "data"
#     if not data_root.exists():
#         logging.error(f"Data root {data_root} does not exist.")
#         sys.exit(1)
    
#     # find 'chunk-*' folders
#     chunk_folders = [f for f in data_root.iterdir() if f.is_dir() and f.name.startswith("chunk-")]
#     if not chunk_folders:
#         logging.error(f"No chunk folders found in {data_root}.")
#         sys.exit(1)

#     for chunk_folder in chunk_folders:
#         # find 'part-*.parquet' files
#         # parquet_files = [str(f) for f in chunk_folder.iterdir() if f.is_file() and f.name.endswith(".parquet")]
#         parquet_files = []
#         for f in chunk_folder.iterdir():
#             if f.is_file() and f.name.endswith(".parquet"):
#                 parquet_files.append(str(f))
#             if len(parquet_files) >= max_files:
#                 break
#         if not parquet_files:
#             logging.error(f"No parquet files found in {chunk_folder}.")
#             continue
#         parquet_paths.extend(parquet_files)
#         if len(parquet_paths) >= max_files:
#             break
    
#     logging.info(f"Found {len(parquet_paths)} parquet files in {dataset_path}.")
#     return parquet_paths

def process_parquet_file(parquet_path: str):
    try:
        df = pd.read_parquet(parquet_path, columns=["bbox_index"])
        check_sum = np.sum(df['bbox_index'] != 0)
        # print(f"Checking {parquet_path}, bbox_index count: {check_sum}")
        # print(check_sum)
        if check_sum > 0:
            return True
        else:
            return False
    except Exception as e:
        return False

def check_meta(dataset_path):
    try:
        meta_dir = Path(dataset_path) / 'meta'
        bbox_json_path = meta_dir / 'bboxes.jsonl'
        if not bbox_json_path.exists():
            return False
    
        records = []
        with open(bbox_json_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        if len(records) < 2:
            return False
    
        if records[0]['bbox_index'] == 0 and records[0]['bbox'] == [] and records[1]['bbox_index'] == 1 and records[1]['bbox'] != []:
            return True

        return False
    
    except Exception as e:
        return False

def check_column(dataset_path):
    all_parquet_files = load_parquet(dataset_path, max_files=10000)
    all_parquet_files = random.sample(all_parquet_files, min(100, len(all_parquet_files)))
    for parquet_path_str in tqdm(all_parquet_files, desc="Checking Parquet Files", unit="file"):
        if process_parquet_file(parquet_path_str):
            return True
    return False
  


if __name__ == "__main__":
    # dataset_home = '/pdata/oxe_lerobot'
    # all_datasets = [str(f) for f in Path(dataset_home).iterdir() if f.is_dir()]
    
    all_datasets = ['/pdata/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/berkeley_fanuc_manipulation', '/pdata/oxe_lerobot/bc_z', '/pdata/oxe_lerobot/nyu_door_opening_surprising_effectiveness', '/pdata/oxe_lerobot/robo_set', '/pdata/oxe_lerobot/ucsd_pick_and_place_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/stanford_hydra_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/iamlab_cmu_pickup_insert_converted_externally_to_rlds', '/pdata/oxe_lerobot/io_ai_tech', '/pdata/oxe_lerobot/cmu_play_fusion', '/pdata/oxe_lerobot/roboturk', '/pdata/oxe_lerobot/fmb', '/pdata/oxe_lerobot/language_table', '/pdata/oxe_lerobot/austin_sailor_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/bridge_data_v2', '/pdata/oxe_lerobot/berkeley_autolab_ur5', '/pdata/oxe_lerobot/bridge', '/pdata/oxe_lerobot/austin_sirius_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/dobbe', '/pdata/oxe_lerobot/cmu_stretch', '/pdata/oxe_lerobot/utaustin_mutex', '/pdata/oxe_lerobot/fractal20220817_data', '/pdata/oxe_lerobot/jaco_play', '/pdata/oxe_lerobot/robo_net', '/pdata/oxe_lerobot/qut_dexterous_manpulation', '/pdata/oxe_lerobot/columbia_cairlab_pusht_real']

    # all_datasets = [
    #     '/pdata/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds', 
    #     '/pdata/oxe_lerobot/berkeley_fanuc_manipulation'
    # ]

    ready_datasets = []
    for dataset_path in all_datasets:
        if check_meta(dataset_path) and check_column(dataset_path):
            print(f"Checking dataset: {dataset_path}, result: True")
            ready_datasets.append(dataset_path)
        else:
            print(f"{dataset_path}: Negative")
    
    if ready_datasets:
        print(ready_datasets)