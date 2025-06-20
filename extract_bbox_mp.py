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

# --- MODIFICATION 1: Remove the global variable ---
# The list of failed files will now be returned by the worker functions.
# failed_parquet_files = [] 


# --- MODIFICATION 2: Modify process_parquet_file to accept a list for failed files ---
# Instead of modifying a global variable, it will append to a list passed as an argument.
def process_parquet_file(parquet_path: str, all_bboxes, local_failed_files: list):
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        # Append to the list provided by the caller (process_dataset)
        print(f"Error reading {parquet_path}: {e}")
        local_failed_files.append(parquet_path)
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
            df.at[index, 'bbox_index'] = 0
        else:
            bbidx = len(all_bboxes)
            df.at[index, 'bbox_index'] = bbidx
            all_bboxes.append({
                'bbox_index': bbidx,
                'bbox': bbox_json
            })
    
    df.to_parquet(parquet_path)


# --- MODIFICATION 3: Modify process_dataset to return the list of failed files ---
def process_dataset(dataset_path):
    # This print might interleave with prints from other processes, which is okay.
    # It shows that multiple datasets are being processed concurrently.
    print(f"Starting to process dataset: {dataset_path}")
    
    # Each process will have its own list of failed files for the dataset it's working on.
    local_failed_parquet_files = []

    json_dir = Path(dataset_path) / 'meta'

    if (json_dir / 'bbox.json').exists():
        os.remove(json_dir / 'bbox.json')
    if (json_dir / 'bbox.jsonl').exists():
        os.remove(json_dir / 'bbox.jsonl')
    if (json_dir / 'bboxes.json').exists():
        os.remove(json_dir / 'bboxes.json')
    if (json_dir / 'bboxes.jsonl').exists():
        os.remove(json_dir / 'bboxes.jsonl')

    all_bboxes = []
    all_bboxes.append({
        'bbox_index': 0,
        'bbox': []
    })
    all_parquet_files = load_parquet(dataset_path)

    # The inner progress bar can be useful to see progress within a single large dataset.
    # We use leave=False so it cleans up after finishing and doesn't clutter the main progress bar.
    dataset_name = Path(dataset_path).name
    for parquet_path_str in tqdm(all_parquet_files, desc=f"Files in {dataset_name}", unit="file", leave=False):
        # Pass the local list to the processing function
        process_parquet_file(parquet_path_str, all_bboxes, local_failed_parquet_files)

    bboxes_json_path = json_dir / "bboxes.jsonl"
    save_list_to_jsonl(all_bboxes, str(bboxes_json_path))
    
    print(f"Finished processing dataset: {dataset_path}")

    # Return the list of files that failed during the processing of this dataset.
    return local_failed_parquet_files


# --- MODIFICATION 4: Import multiprocessing and create the main parallel execution block ---
import multiprocessing

if __name__ == "__main__":
    # It's crucial to put the main execution logic inside `if __name__ == "__main__":`
    # when using multiprocessing to prevent issues on some platforms (like Windows).

    all_datasets = ['/pdata/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/bc_z', '/pdata/oxe_lerobot/nyu_door_opening_surprising_effectiveness', '/pdata/oxe_lerobot/robo_set', '/pdata/oxe_lerobot/ucsd_pick_and_place_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/stanford_hydra_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/iamlab_cmu_pickup_insert_converted_externally_to_rlds', '/pdata/oxe_lerobot/io_ai_tech', '/pdata/oxe_lerobot/cmu_play_fusion', '/pdata/oxe_lerobot/roboturk', '/pdata/oxe_lerobot/fmb', '/pdata/oxe_lerobot/language_table', '/pdata/oxe_lerobot/austin_sailor_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/bridge_data_v2', '/pdata/oxe_lerobot/berkeley_autolab_ur5', '/pdata/oxe_lerobot/bridge', '/pdata/oxe_lerobot/austin_sirius_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/dobbe', '/pdata/oxe_lerobot/cmu_stretch', '/pdata/oxe_lerobot/utaustin_mutex', '/pdata/oxe_lerobot/fractal20220817_data', '/pdata/oxe_lerobot/jaco_play', '/pdata/oxe_lerobot/robo_net', '/pdata/oxe_lerobot/qut_dexterous_manpulation', '/pdata/oxe_lerobot/columbia_cairlab_pusht_real']
    already_checked_datasets = [
        '/pdata/oxe_lerobot/bc_z', '/pdata/oxe_lerobot/nyu_door_opening_surprising_effectiveness',
        '/pdata/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds', '/pdata/oxe_lerobot/berkeley_fanuc_manipulation',
    ]
    all_datasets = [d for d in all_datasets if d not in already_checked_datasets]

    # This will be the main list in the master process to collect all failures.
    all_failed_parquet_files = []
    
    # Determine the number of parallel workers. os.cpu_count() is a good default.
    # You can set this to a specific number, e.g., 8, if you want.
    # num_workers = os.cpu_count() or 4
    num_workers = 20
    print(f"Starting parallel processing with {num_workers} workers for {len(all_datasets)} datasets.")

    # Create a process pool. The `with` statement ensures the pool is properly closed.
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use pool.imap_unordered to apply the function to each dataset path.
        # This function processes the list of datasets in parallel.
        # tqdm is wrapped around the iterator to create a progress bar for datasets.
        results_iterator = pool.imap_unordered(process_dataset, all_datasets)
        
        progress_bar = tqdm(results_iterator, total=len(all_datasets), desc="Overall Dataset Progress")

        # As each process finishes its task, its return value (the list of failed files) is yielded.
        for failed_list_from_one_dataset in progress_bar:
            if failed_list_from_one_dataset:
                # Extend the main list with the failures from this dataset.
                all_failed_parquet_files.extend(failed_list_from_one_dataset)

    print("\nAll datasets have been processed.")

    # Finally, report and save the collected list of all failed files.
    if all_failed_parquet_files:
        print("\nFailed to read the following Parquet files across all datasets:")
        # Using set to print only unique file paths in case of any duplication
        for f in sorted(list(set(all_failed_parquet_files))):
            print(f)
        
        with open('failed_parquet_files.pkl', 'wb') as f:
            pickle.dump(all_failed_parquet_files, f)
        print("\nList of failed files saved to 'failed_parquet_files.pkl'")
    else:
        print("\nSuccessfully processed all datasets with zero read failures.")