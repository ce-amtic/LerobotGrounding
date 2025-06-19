import time
import json
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from batched_grounding import load_parquet
from qwen_grounding_utils_v3 import load_list_from_jsonl, normalize_bbox, check_normalized

def dump_record_to_finish_json(new_record):
    with open("finished_parquets.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(new_record, ensure_ascii=False) + "\n")

finished_parquet_list = load_list_from_jsonl("finished_parquets.jsonl")
finished_parquet_list = [item["path"] for item in finished_parquet_list]

def process_parquet_file(parquet_path: str):
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading {parquet_path}: {e}")
        return

    if 'bbox' in df.columns:
        dump_record_to_finish_json({"path": parquet_path})
        logging.info(f"Marked {parquet_path} as it already has 'bbox' column.")

def process_dataset(dataset_path):
    global finished_parquet_list
    print(f"Processing dataset: {dataset_path}")

    all_parquet_files = load_parquet(dataset_path)
    for parquet_path_str in tqdm(all_parquet_files, desc="Extracting from Parquet Files", unit="file"):
        if parquet_path_str in finished_parquet_list:
            continue
        process_parquet_file(parquet_path_str)
 

if __name__ == "__main__":
    dataset_home = '/pdata/oxe_lerobot'
    all_datasets = [str(f) for f in Path(dataset_home).iterdir() if f.is_dir()]

    dump_record_to_finish_json({
        "path": str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    })

    for i, dataset_path in enumerate(all_datasets):
        # if dataset_path.endswith('bc_z'):
        #     print(f"Skipping dataset {i+1}/{len(all_datasets)}: {dataset_path} (bc_z)")
        #     continue

        print(f"Processing dataset {i+1}/{len(all_datasets)}: {dataset_path}")
        process_dataset(dataset_path)
        print(f"Finished processing dataset {i+1}/{len(all_datasets)}: {dataset_path}")