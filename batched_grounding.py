"""
CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=true python batched_grounding.py \
    --dataset-path /pdata/oxe_lerobot/dlr_sara_pour_converted_externally_to_rlds \
    --output-home /pdata/oxe_lerobot_g \
    --batch_size 24

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false python batched_grounding.py \
    --dataset-home /pdata/oxe_lerobot \
    --inplace 1 \
    --batch-size 1024
"""

import os, sys, io, signal, time
import argparse
import logging
import pandas
import shutil
import random
import json
import atexit
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Dataset, Value
from datasets.features.image import Image as ImageHF
from qwen_grounding_utils_v3 import setup_model, grounding_pipeline_batched

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# set DEBUG=1 to enable debug logs
if os.getenv("DEBUG"):
    logging.getLogger().setLevel(logging.DEBUG)

### --- global variables --- ###
SHOW_OBJECT_NAME = False

USE_SUBTASK_CONDITIONING = True

# SAVE_INSPECTION_IMAGE = True
# RANDOM_SAMPLE_RATE = 0.0005 # ratio of images that will be saved for inspection

OVERWRITE_ORIGINAL_DATASET = False

# MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME = "/fyh/.env/models/Qwen2.5-VL-7B-Instruct"
BATCH_SIZE = 8


### --- global varibles --- ###
from qwen_grounding_utils_v3 import load_list_from_jsonl, save_list_to_jsonl

finished_parquet_list = load_list_from_jsonl("finished_parquets.jsonl")
finished_parquet_list = [item["path"] for item in finished_parquet_list]
def save_finish_list_to_json():
    global finished_parquet_list
    finished_parquets = []
    for item in finished_parquet_list:
        if isinstance(item, str):
            finished_parquets.append({"path": item})
    save_list_to_jsonl(finished_parquets, "finished_parquets.jsonl")

# failed_dataset_list = load_list_from_jsonl("failed_datasets.jsonl")
# def save_fail_list_to_json():
#     global failed_dataset_list
#     logging.info(f"There are {len(failed_dataset_list)} datasets failed to process. The paths and features will be saved to './failed_datasets.jsonl'.")
#     save_list_to_jsonl(failed_dataset_list, "failed_datasets.jsonl")

def handle_sigint(signum, frame):
    logging.info("Caught Ctrl+C, saving list before exit...")
    save_finish_list_to_json()
    # save_fail_list_to_json()
    sys.exit(0)

inspection_image_id = 0
os.makedirs("inspection_images", exist_ok=True)

q_model, q_processor = None, None


### --- process parquet --- ###
def load_parquet(dataset_path):
    parquet_paths = []

    if type(dataset_path) is str:
        dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logging.error(f"Dataset in {dataset_path} does not exist.")
        sys.exit(1)
    
    data_root = dataset_path / "data"
    if not data_root.exists():
        logging.error(f"Data root {data_root} does not exist.")
        sys.exit(1)
    
    # find 'chunk-*' folders
    chunk_folders = [f for f in data_root.iterdir() if f.is_dir() and f.name.startswith("chunk-")]
    if not chunk_folders:
        logging.error(f"No chunk folders found in {data_root}.")
        sys.exit(1)

    for chunk_folder in chunk_folders:
        # find 'part-*.parquet' files
        parquet_files = [str(f) for f in chunk_folder.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        if not parquet_files:
            logging.error(f"No parquet files found in {chunk_folder}.")
            continue
        parquet_paths.extend(parquet_files)
    
    logging.info(f"Found {len(parquet_paths)} parquet files in {dataset_path}.")
    return parquet_paths

def _save_and_clear_file_from_memory(
    path_str,
    in_memory_data,
    original_features_map
):
    """
    Saves a single, fully processed Parquet file to disk and removes its data
    from memory to improve efficiency.
    """
    global finished_parquet_list
    logging.info(f"All images for {path_str} processed. Saving and clearing from memory.")
    
    updated_records = in_memory_data.get(path_str)
    if not updated_records:
        logging.warning(f"Attempted to save {path_str}, but no records found in memory. Skipping.")
        return

    try:
        original_features = original_features_map[path_str]
        new_features = original_features.copy()
        
        if 'bbox' not in new_features:
            new_features['bbox'] = Value("string")
        
        updated_dataset = Dataset.from_list(updated_records, features=new_features)
        updated_dataset.to_parquet(path_str)
        
        # Mark as finished ONLY after successful save
        finished_parquet_list.append(path_str)
        logging.info(f"Successfully saved {path_str}")

    except Exception as e:
        logging.error(f"Error saving updated dataset to {path_str}: {e}. It will remain in memory for final save attempt.")
        # We DON'T remove it from memory if save fails, allowing a final attempt later.
        return

    # --- Cleanup ---
    # If save was successful, remove data from all tracking dictionaries.
    del in_memory_data[path_str]
    del original_features_map[path_str]

def _process_and_apply_batch(
    images,
    metadata,
    task_descs,
    in_memory_data,
    all_episode_bboxes_data,
    original_features_map,
    *, # Keyword-only arguments below
    model,
    processor,
    dataset_tag,
    show_object_name,
    use_subtask_conditioning
):
    """
    Helper function to process a batch of images and apply the results.
    This function modifies `in_memory_data` and `all_episode_bboxes_data` in place.
    """
    global inspection_image_id
    logging.debug(f"Processing batch of size {len(images)}")

    batched_json_responses, _ = grounding_pipeline_batched(
        images=images,
        model=model,
        processor=processor,
        task_descs=task_descs,
        SHOW_OBJECT_NAME=show_object_name,
        USE_SUBTASK_CONDITIONING=use_subtask_conditioning
    )

    # Distribute results back to the correct records in memory
    prevpath = None
    for i in range(len(batched_json_responses)):
        meta = metadata[i]
        path = meta["original_parquet_path"]
        r_idx = meta["original_record_idx"]
        json_response = batched_json_responses[i]

        if prevpath is not None and prevpath != path:
            # Save and clear the previous file from memory
            _save_and_clear_file_from_memory(
                prevpath,
                in_memory_data,
                original_features_map
            )
            save_finish_list_to_json()
        prevpath = path

        # Get the specific record from our in-memory store
        record_to_update = in_memory_data[path][r_idx]

        # --- Save inspection image logic (unchanged) ---
        # if SAVE_INSPECTION_IMAGE and (random.random() < RANDOM_SAMPLE_RATE):
        #     inspection_image_id += 1
        #     parquet_info_str = f"{dataset_tag}_{Path(path).stem}_{meta['image_col_name'][-1]}".replace('/', '-').replace('\\', '-')
        #     save_json_infos = {
        #         'parquet_path': path,
        #         'record_idx': r_idx,
        #         'image_col_name': meta['image_col_name'],
        #         'task_desc': meta['task_str_used'],
        #         'json_response': json_response
        #     }
        #     with open(f"inspection_images/{inspection_image_id}_{parquet_info_str}.json", 'w') as json_file:
        #         json.dump(save_json_infos, json_file, ensure_ascii=False, indent=4)

        # --- Update the record with bbox info ---
        if 'bbox' not in record_to_update:
            record_to_update['bbox'] = json.dumps(json_response, ensure_ascii=False)
        
            # Update the separate bboxes.jsonl file as well
            bbox_idx = len(all_episode_bboxes_data)
            all_episode_bboxes_data.append({
                'bbox_index': bbox_idx,
                'bbox': json_response
            })

def process_parquet(parquet_paths, tasks=None, bboxes_json_path: Path = None, dataset_tag=""):
    """
    Processes a list of Parquet files, batching images across files for efficiency,
    and then writes the updated data back to each file.
    """
    global finished_parquet_list
    # global failed_dataset_list

    # Load shared bbox data if it exists
    if bboxes_json_path and bboxes_json_path.exists():
        all_episode_bboxes_data = load_list_from_jsonl(str(bboxes_json_path))
    else:
        all_episode_bboxes_data = []

    tasks_available = USE_SUBTASK_CONDITIONING and tasks is not None
    if USE_SUBTASK_CONDITIONING and tasks is None:
        logging.warning("No tasks provided. Disabling subtask conditioning for all files.")

    # --- Data structures for cross-file batching ---
    # Batches that will accumulate data across multiple parquet files
    current_batch_images = []
    current_batch_task_descs = []
    current_batch_metadata = []

    # In-memory storage for all records from all files to be processed
    # Key: str(parquet_path), Value: list of record dictionaries
    in_memory_data = {}
    original_features_map = {}

    # =========================================================================
    # PHASE 1: DATA COLLECTION AND BATCH PROCESSING
    # =========================================================================
    logging.info("Phase 1: Collecting images and processing batches...")
    for parquet_path_str in tqdm(parquet_paths, desc="Collecting from Parquet Files", unit="file"):
        parquet_path = Path(parquet_path_str)
        if str(parquet_path) in finished_parquet_list:
            logging.info(f"Skipping already processed file: {parquet_path}")
            continue

        try:
            episode = load_dataset("parquet", data_files=str(parquet_path), split='train')
            features = episode.features
            logging.debug(f"Loaded {parquet_path} with {len(episode)} records.")
        except Exception as e:
            logging.error(f"Error loading {parquet_path}: {e}. Skipping file.")
            # failed_dataset_list.append({"path": str(parquet_path), "reason": f"Load error: {e}"})
            continue # Skip to the next file

        if 'sub_task_index' not in features:
            logging.info(f"Skipping {parquet_path} because it lacks 'sub_task_index' column.")
            continue
        
        # Store records and features in our in-memory maps
        temp_episode_records = list(episode)
        in_memory_data[str(parquet_path)] = temp_episode_records
        original_features_map[str(parquet_path)] = features

        # --- Identify image columns ---
        image_columns_info = []
        suspected_image_columns = [col for col in features if col == "observation.images.cam"]
        for col in suspected_image_columns:
            if isinstance(features[col], ImageHF):
                image_columns_info.append((col, ImageHF))
            elif isinstance(features[col], dict) and 'bytes' in features[col]:
                image_columns_info.append((col, dict))

        if not image_columns_info:
            logging.error(f"No recognized image columns in {parquet_path}. Skipping.")
            del in_memory_data[str(parquet_path)]
            del original_features_map[str(parquet_path)]
            continue

        # --- Iterate records to fill batches ---
        for r_idx, record in enumerate(temp_episode_records):
            if record.get('sub_task_index', -1) == -1:
                continue

            # --- Task description logic ---
            task_str = None
            final_use_task_for_record = tasks_available
            if final_use_task_for_record:
                r_task_index = record.get('task_index', None)
                if 'task_index' not in features:
                    if r_idx == 0: logging.warning(f"No task_index column in {parquet_path}. Disabling subtask conditioning for this file.")
                    final_use_task_for_record = False
                elif r_task_index is not None:
                    try:
                        # task_row = tasks[tasks['task_index'] == r_task_index]
                        # task_str = task_row.iloc[0]['task'] if not task_row.empty else None
                        task_str = tasks[int(r_task_index)]
                        if task_str is None:
                            final_use_task_for_record = False
                            logging.warning(f"Task_index {r_task_index} not in tasks.jsonl for {parquet_path}, record {r_idx}.")
                    except Exception as e:
                        logging.warning(f"Error retrieving task for {r_task_index} in {parquet_path}, record {r_idx}: {e}.")
                        final_use_task_for_record = False
                else:
                    final_use_task_for_record = False

            # --- Add images from this record to the batch ---
            for image_col_name, image_col_type in image_columns_info:
                try:
                    pil_image = None
                    if image_col_type == ImageHF:
                        if record[image_col_name] is not None:
                            pil_image = record[image_col_name]
                    elif image_col_type == dict:
                        img_bytes_data = record[image_col_name]
                        if img_bytes_data and img_bytes_data.get('bytes'):
                            pil_image = Image.open(io.BytesIO(img_bytes_data['bytes']))
                    
                    if pil_image is None:
                        logging.warning(f"Image is None for record {r_idx}, col {image_col_name} in {parquet_path}. Skipping image.")
                        continue
                    
                    # Add to cross-file batch
                    current_batch_images.append(pil_image)
                    current_batch_task_descs.append(task_str if final_use_task_for_record else None)
                    current_batch_metadata.append({
                        "original_parquet_path": str(parquet_path), # CRITICAL: Store which file this came from
                        "original_record_idx": r_idx,
                        "image_col_name": image_col_name,
                        "task_str_used": task_str if final_use_task_for_record else None
                    })
                except Exception as e:
                    logging.error(f"Error preparing image from record {r_idx}, col {image_col_name} in {parquet_path}: {e}")
                    continue

                # --- If batch is full, process it ---
                if len(current_batch_images) >= BATCH_SIZE:
                    _process_and_apply_batch(
                        current_batch_images, current_batch_metadata, current_batch_task_descs,
                        in_memory_data, all_episode_bboxes_data, original_features_map,
                        model=q_model, processor=q_processor, dataset_tag=dataset_tag,
                        show_object_name=SHOW_OBJECT_NAME, use_subtask_conditioning=USE_SUBTASK_CONDITIONING
                    )
                    # Clear batches for the next round
                    current_batch_images, current_batch_task_descs, current_batch_metadata = [], [], []

    # --- Process the final remaining batch ---
    if current_batch_images:
        logging.info(f"Processing final batch of size {len(current_batch_images)}.")
        _process_and_apply_batch(
            current_batch_images, current_batch_metadata, current_batch_task_descs,
            in_memory_data, all_episode_bboxes_data, original_features_map,
            model=q_model, processor=q_processor, dataset_tag=dataset_tag,
            show_object_name=SHOW_OBJECT_NAME, use_subtask_conditioning=USE_SUBTASK_CONDITIONING
        )

    # =========================================================================
    # PHASE 2: WRITING UPDATED DATA BACK TO FILES
    # =========================================================================
    logging.info("Phase 2: Saving updated parquet files...")
    if in_memory_data:
        logging.warning(f"Saving {len(in_memory_data)} file(s) that remained in memory.")
        remaining_paths = list(in_memory_data.keys())
        for path_str in tqdm(remaining_paths, desc="Saving Remaining Files", unit="file"):
            if path_str in in_memory_data: # Check if it wasn't already processed by a concurrent call
                _save_and_clear_file_from_memory(
                    path_str, in_memory_data, original_features_map
                )
        save_finish_list_to_json()

    # --- Final save for shared data and progress tracking ---
    try:
        if bboxes_json_path:
            save_list_to_jsonl(all_episode_bboxes_data, str(bboxes_json_path))
            logging.info(f"Updated {bboxes_json_path} with {len(all_episode_bboxes_data)} total items.")
        save_finish_list_to_json()
    except Exception as e:
        logging.error(f"Error saving final JSONL/finish list files: {e}")
        return False

    logging.info("All files processed successfully.")
    return True

def process_dataset(dataset_path: str):
    if not OVERWRITE_ORIGINAL_DATASET:
        # copy dataset folder to output home
        if not args.output_home:
            logging.error("Output home directory is not specified. Please provide --output-home.")
            sys.exit(1)
        output_dataset_path = Path(args.output_home) / Path(dataset_path).name
        try:
            if output_dataset_path.exists():
                logging.info(f"Output dataset path {output_dataset_path} already exists. Removing it...")
                shutil.rmtree(output_dataset_path)
            logging.info(f"Copying dataset {Path(dataset_path).name} to {output_dataset_path}...")
            shutil.copytree(dataset_path, output_dataset_path)
            logging.info(f"Dataset {Path(dataset_path).name} copied to {output_dataset_path}")
        except Exception as e:
            logging.error(f"Failed to copy {Path(dataset_path).name}: {e}")
    else:
        output_dataset_path = Path(dataset_path)

    # load tasks.jsonl
    task_json_path = output_dataset_path / "meta" / "tasks.jsonl"
    if task_json_path.exists():
        tasks_df = pandas.read_json(task_json_path, lines=True)
        tasks = {int(row['task_index']): row['task'] for _, row in tasks_df.iterrows()}
    else:
        tasks = None

    # process each parquet file
    finished = process_parquet(
        load_parquet(output_dataset_path), 
        tasks=tasks, 
        bboxes_json_path=output_dataset_path / "meta" / "bboxes.jsonl",
        dataset_tag=output_dataset_path.name[:6]
    )
    if not finished:
        logging.error(f"Failed to process dataset {dataset_path}. Please check the logs for details.")
        return

    logging.info(f"Processed dataset {dataset_path} successfully.")


def main(args):
    if args.dataset_path:
        dataset_path = args.dataset_path
        process_dataset(dataset_path)
    elif args.dataset_home:
        dataset_home = args.dataset_home
        all_datasets = [str(f) for f in Path(dataset_home).iterdir() if f.is_dir()]
        logging.info(f"Found {len(all_datasets)} datasets in {dataset_home}.")
        for i, dataset_path in enumerate(all_datasets):
            logging.info(f"Processing {i+1} out of {len(all_datasets)}.")
            process_dataset(dataset_path)
    else:
        logging.error("Please provide either --dataset-path or --dataset-home.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate through a LeRobot dataset.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path of the LeRobot dataset (e.g., '~/lerobot/pusht').",
    )
    parser.add_argument(
        "--dataset-home",
        type=str,
        help="Home directory of all datasets (e.g., '~/lerobot').",
    )
    parser.add_argument(
        "--output-home",
        type=str,
        help="Output home directory for processed data (optional).",
    )
    parser.add_argument(
        "--inplace",
        type=int,
        default=0,
        choices=[0, 1],
        help="If set to 1, will overwrite the original dataset with the processed data.",
    )
    parser.add_argument(
        "--finished-list-path",
        type=str,
        help="Path to finish_parquets.json, required when inplace==1."   
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for processing (default: {BATCH_SIZE}).",
    )
    args = parser.parse_args()

    finished_parquet_list.append(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    # failed_dataset_list.append({"time": str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))})
    
    if args.inplace == 1:
        OVERWRITE_ORIGINAL_DATASET = True
        logging.warning("Output home directory is not specified. Will overwrite the original dataset.")
    
    atexit.register(save_finish_list_to_json)
    # atexit.register(save_fail_list_to_json)
    signal.signal(signal.SIGINT, handle_sigint)
    
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    logging.info(f"Batch size set to {BATCH_SIZE}.")

    q_model, q_processor = setup_model(MODEL_NAME)
    
    main(args)