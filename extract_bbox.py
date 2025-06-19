import os, io
import json
import logging
import pickle
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from batched_grounding import load_parquet
from qwen_grounding_utils_v3 import save_list_to_jsonl, normalize_bbox, check_normalized

failed_parquet_files = []

def process_parquet_file(parquet_path:str, all_bboxes):
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        global failed_parquet_files

        print(f"Error reading {parquet_path}: {e}")
        failed_parquet_files.append(parquet_path)
        # raise NotImplementedError(f"Failed to read Parquet file: {parquet_path}")
        return

    if 'bbox' not in df.columns:
        print(f"Warning: 'bbox' column not found in {parquet_path}. Skipping this file.")
        return

    for index, row in df.iterrows():
        bbox_str = row['bbox']

        parse_ok = False
        if bbox_str is not None:
            try:
                bbox_json = json.loads(bbox_str)
                assert isinstance(bbox_json, list), "bbox should be a list"
                assert bbox_json, "bbox list should not be empty"

                if not check_normalized(bbox_json):
                    img_data_bytes = row["observation.images.cam"]
                    if img_data_bytes and isinstance(img_data_bytes, bytes):
                        pil_img = Image.open(io.BytesIO(img_data_bytes))
                    elif img_data_bytes and isinstance(img_data_bytes, dict) and 'bytes' in img_data_bytes:
                        pil_img = Image.open(io.BytesIO(img_data_bytes['bytes']))
                    else:
                        raise NotImplementedError(f"Unsupported image data type: {type(img_data_bytes)} in {parquet_path} at index {index}")
                    width, height = pil_img.size
                    bbox_json = normalize_bbox(bbox_json, width, height)

                parse_ok = True
            except Exception as e:
                # logging.error(f"Error parsing bbox in {parquet_path} at index {index}: {e}")
                pass
        
        if not parse_ok:
            row['bbox_index'] = 0
        else:
            row['bbox_index'] = len(all_bboxes)
            all_bboxes.append({
                'bbox_index': row['bbox_index'],
                'bbox': bbox_json
            })
    
    df.to_parquet(parquet_path)
    

def process_dataset(dataset_path):
    print(f"Processing dataset: {dataset_path}")
    json_dir = Path(dataset_path) / 'meta'

    if (json_dir / 'bbox.json').exists():
        os.remove(json_dir / 'bbox.json')
    if (json_dir / 'bbox.jsonl').exists():
        os.remove(json_dir / 'bbox.jsonl')
    if (json_dir / 'bboxes.json').exists():
        os.remove(json_dir / 'bboxes.json')
    if (json_dir / 'bboxes.jsonl').exists():
        os.remove(json_dir / 'bboxes.jsonl')
    # if (json_dir / 'bbox_rewrite.jsonl').exists():
    #     os.remove(json_dir / 'bbox_rewrite.jsonl')

    all_bboxes = []
    all_bboxes.append({
        'bbox_index': 0,
        'bbox': []
    })
    all_parquet_files = load_parquet(dataset_path)

    for parquet_path_str in tqdm(all_parquet_files, desc="Extracting from Parquet Files", unit="file"):
        process_parquet_file(parquet_path_str, all_bboxes)

    bboxes_json_path = json_dir / "bboxes.jsonl"
    save_list_to_jsonl(all_bboxes, str(bboxes_json_path))    


if __name__ == "__main__":
    # dataset_home = '/pdata/oxe_lerobot'
    # all_datasets = [str(f) for f in Path(dataset_home).iterdir() if f.is_dir()]

    # all_datasets = ['/pdata/oxe_lerobot/bc_z', '/pdata/oxe_lerobot/nyu_door_opening_surprising_effectiveness', '/pdata/oxe_lerobot/robo_set', '/pdata/oxe_lerobot/ucsd_pick_and_place_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/stanford_hydra_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/iamlab_cmu_pickup_insert_converted_externally_to_rlds', '/pdata/oxe_lerobot/io_ai_tech', '/pdata/oxe_lerobot/cmu_play_fusion', '/pdata/oxe_lerobot/roboturk', '/pdata/oxe_lerobot/fmb', '/pdata/oxe_lerobot/language_table']
    all_datasets = [
        '/pdata/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds'
    ]

    for i, dataset_path in enumerate(all_datasets):
        # if dataset_path.endswith('bc_z'):
        #     print(f"Skipping dataset {i+1}/{len(all_datasets)}: {dataset_path} (bc_z)")
        #     continue

        print(f"Processing dataset {i+1}/{len(all_datasets)}: {dataset_path}")
        process_dataset(dataset_path)
        print(f"Finished processing dataset {i+1}/{len(all_datasets)}: {dataset_path}")
    
    if failed_parquet_files:
        print("Failed to read the following Parquet files:")
        print("\n".join(failed_parquet_files))
        pickle.dump(failed_parquet_files, open('failed_parquet_files.pkl', 'wb'))