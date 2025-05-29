"""
CUDA_VISIBLE_DEVICES=6,7 TOKENIZERS_PARALLELISM=false python grounding_v4.py \
    --dataset-path /pdata/oxe_lerobot/dlr_sara_pour_converted_externally_to_rlds \
    --output-home /pdata/oxe_lerobot_g \
    --num-proc 8 > o.txt

CUDA_VISIBLE_DEVICES=6,7 TOKENIZERS_PARALLELISM=false python grounding_v4.py \
    --dataset-home /pdata/oxe_lerobot \
    --finished-list-path finished_parquets.json \
    --inplace 1 \
    --num-proc 8 > o.txt
"""

import os, sys, io, signal
import argparse
import logging
import pandas
import shutil
import random
import json
import atexit
import threading
import torch # For CUDA device management
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Dataset, Value, Features
from datasets.features.image import Image as ImageHF
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
# set DEBUG=1 to enable debug logs
if os.getenv("DEBUG"):
    logging.getLogger().setLevel(logging.DEBUG)

### --- global constants --- ###
SHOW_OBJECT_NAME = False
USE_SUBTASK_CONDITIONING_GLOBAL = True # Global toggle
SAVE_INSPECTION_IMAGE = True
RANDOM_SAMPLE_RATE = 0.0005 # ratio of images that will be saved for inspection
# OVERWRITE_ORIGINAL_DATASET is set based on args.inplace
MODEL_NAME = "/fyh/.env/models/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct" # Use this if downloading

### --- qwen pipeline --- ###
from qwen_grounding_utils import setup_model, grounding_pipeline

### --- Global Shared State (managed by GlobalStateManager) --- ###
class GlobalStateManager:
    def __init__(self):
        self.inspection_image_id = 0
        self.inspection_image_id_lock = threading.Lock()
        
        self.failed_dataset_list = []
        self.failed_dataset_list_lock = threading.Lock()
        
        self.finished_parquet_list = None # Populated if inplace=1
        self.finished_parquet_list_lock = threading.Lock() # Used if inplace=1
        self.args_finished_list_path = None # For saving path
        self.OVERWRITE_ORIGINAL_DATASET = False

        # For ThreadPoolExecutor initializer
        self.thread_local_models = {} # Stores (q_model, q_processor) per thread_id
        self.worker_device_map = {} # Maps thread_id to cuda_device_id
        # self.next_device_idx = 0
        self.device_assignment_lock = threading.Lock()
        # self.available_cuda_devices = [] # Populated in main
        self.pytorch_device_indices = []
        self.next_pytorch_device_map_idx = 0

global_state = GlobalStateManager()
executor_ref = None 

### --- Worker Initialization --- ###
def init_worker_model_on_thread():
    thread_id = threading.get_ident()
    
    with global_state.device_assignment_lock:
        # This check ensures model is initialized only once if initializer is called multiple times for the same thread
        if thread_id not in global_state.thread_local_models:
            # device_idx_to_use = global_state.next_device_idx
            pytorch_device_ordinal_to_use = global_state.pytorch_device_indices[global_state.next_pytorch_device_map_idx]
            global_state.next_pytorch_device_map_idx = (global_state.next_pytorch_device_map_idx + 1) % len(global_state.pytorch_device_indices)
            # global_state.next_device_idx = (global_state.next_device_idx + 1) % len(global_state.available_cuda_devices)
            # actual_device_id = global_state.available_cuda_devices[device_idx_to_use]
            global_state.worker_device_map[thread_id] = pytorch_device_ordinal_to_use
            
            torch.cuda.set_device(pytorch_device_ordinal_to_use) 
            logging.info(f"Worker (thread {thread_id}) on device cuda:{pytorch_device_ordinal_to_use} initializing model {MODEL_NAME}...")
            q_model, q_processor = setup_model(MODEL_NAME)
            global_state.thread_local_models[thread_id] = (q_model, q_processor)
            logging.info(f"Worker (thread {thread_id}) on device cuda:{pytorch_device_ordinal_to_use} model initialized.")

### --- Core Parquet Processing Task (Image Processing Stage) --- ###
def process_parquet_task(job_args):
    (parquet_path_str, tasks_df_json, output_dataset_path_str,
     use_subtask_conditioning_config, show_object_name_config,
     save_inspection_image_config, random_sample_rate_config) = job_args

    thread_id = threading.get_ident()
    q_model, q_processor = global_state.thread_local_models[thread_id]
    worker_device = global_state.worker_device_map[thread_id]
    
    parquet_path = Path(parquet_path_str)
    tasks_df = pandas.read_json(io.StringIO(tasks_df_json), lines=True) if tasks_df_json else None

    processed_records_for_this_file = []
    r_bboxes_for_records = []
    original_features_schema_dict = None
    
    try:
        episode = load_dataset("parquet", data_files=str(parquet_path), split='train', cache_dir=f"./.cache/datasets_cache_w{worker_device}")
        features = episode.features
        original_features_schema_dict = features.to_dict() # Save schema for reconstruction
        logging.debug(f"Loaded dataset from {parquet_path} with {len(episode)} records by worker on device {worker_device}.")

        image_columns = []
        suspected_image_columns = [col for col in features if col.startswith("observation.images.cam")]
        for col in suspected_image_columns:
            if isinstance(features[col], ImageHF):
                image_columns.append((col, ImageHF))
            elif isinstance(features[col], dict) and 'bytes' in features[col]:
                image_columns.append((col, dict))
            else:
                logging.error(f"Unrecognized image type for column {col}: {type(features[col])} in {parquet_path}")
        
        if not image_columns:
            logging.error(f"Failed to auto-detect image columns in {parquet_path}.")
            return str(parquet_path), original_features_schema_dict, [], [], True, f"No image columns found in {parquet_path}"

        tasks_available_for_this_parquet = (tasks_df is not None)
        task_index_column_present = 'task_index' in features

        # tqdm for records can be noisy, using logging.debug or disable for threads
        # for r_idx, record in tqdm(enumerate(episode), desc=f"Records in {parquet_path.name}", leave=False, position=worker_id % num_proc):
        for r_idx, record in enumerate(episode):
            r_task_index = record.get('task_index', None)
            task_str = None
            task_lookup_successful = True

            # Determine if subtask conditioning should be used for this record
            use_subtask_for_record = use_subtask_conditioning_config and \
                                     tasks_available_for_this_parquet and \
                                     task_index_column_present

            if use_subtask_for_record:
                if r_task_index is not None:
                    try:
                        task_row = tasks_df[tasks_df['task_index'] == r_task_index]
                        if not task_row.empty:
                            task_str = task_row.iloc[0]['task']
                        else:
                            # Log if specific task_index not found, but don't disable for whole parquet unless it's a pattern
                            logging.debug(f"Task_index {r_task_index} not found in tasks_df for {parquet_path}, record {r_idx}.")
                            task_lookup_successful = False 
                    except Exception as e:
                        logging.warning(f"Error retrieving task for task_index {r_task_index} in {parquet_path}, record {r_idx}: {e}. Disabling for this record.")
                        task_lookup_successful = False
                else: # r_task_index is None
                    task_lookup_successful = False
            
            final_use_task_for_record = use_subtask_for_record and task_lookup_successful
            
            r_bbox_json = None # Combined bbox info for the record (from first image usually)
            
            # Create a mutable copy of the record to update image bytes
            updated_record = record.copy()

            for image_col, image_type in image_columns:
                logging.debug(f"Processing record[{r_idx}] in {parquet_path}, image column: {image_col}")

                if image_type == ImageHF:
                    image_pil = record[image_col] # This is already a PIL Image
                elif image_type == dict:
                    image_pil = Image.open(io.BytesIO(record[image_col]['bytes']))
                
                if image_pil is None:
                    logging.warning(f"Could not load image for {image_col} in record {r_idx} of {parquet_path}")
                    continue

                json_response, output_image_pil = grounding_pipeline(
                    image=image_pil,
                    model=q_model,
                    processor=q_processor,
                    task_desc=task_str if final_use_task_for_record else None,
                    SHOW_OBJECT_NAME=show_object_name_config,
                    USE_SUBTASK_CONDITIONING=final_use_task_for_record # Pass the final decision for this record
                )

                if save_inspection_image_config and (random.random() < random_sample_rate_config):
                    with global_state.inspection_image_id_lock:
                        global_state.inspection_image_id += 1
                        current_inspection_id = global_state.inspection_image_id
                    
                    parquet_info_str = str(parquet_path).replace('/', '-').replace('\\', '-').replace('.', '-')
                    inspection_img_path = f"inspection_images/{current_inspection_id}_{parquet_info_str}_{image_col.replace('.', '_')}.jpg"
                    output_image_pil.save(inspection_img_path)
                    
                    save_json_infos = {
                        'parquet_path': str(parquet_path),
                        'record_index': r_idx,
                        'image_column': image_col,
                        'task_desc': task_str if final_use_task_for_record else '',
                        'json_response': json_response
                    }
                    with open(f"inspection_images/{current_inspection_id}_{parquet_info_str}_{image_col.replace('.', '_')}.json", 'w') as json_file:
                        json.dump(save_json_infos, json_file, ensure_ascii=False, indent=4)
                
                # Update the record with the new image bytes
                if image_type == ImageHF:
                    updated_record[image_col] = output_image_pil # datasets library handles PIL conversion
                elif image_type == dict:
                    buffer = io.BytesIO()
                    # Ensure format is preserved or defaults to PNG
                    img_format = output_image_pil.format if output_image_pil.format else "PNG"
                    if img_format == "JPEG" and output_image_pil.mode == "RGBA": # JPEG doesn't support alpha
                        output_image_pil = output_image_pil.convert("RGB")
                    output_image_pil.save(buffer, format=img_format)
                    updated_record[image_col]['bytes'] = buffer.getvalue()
                
                if r_bbox_json is None: # Take bbox from the first successfully processed image in the record
                    r_bbox_json = json_response
            
            processed_records_for_this_file.append(updated_record)
            r_bboxes_for_records.append(r_bbox_json if r_bbox_json is not None else {}) # Ensure valid JSON

        return str(parquet_path), original_features_schema_dict, processed_records_for_this_file, r_bboxes_for_records, False, None

    except Exception as e:
        logging.error(f"Error processing parquet {parquet_path} by worker on device {worker_device}: {e}", exc_info=True)
        return str(parquet_path), original_features_schema_dict, [], [], True, str(e)


### --- utility function to load parquet paths --- ###
def load_parquet_paths_from_dataset(dataset_dir_path: Path):
    parquet_paths = []
    if not dataset_dir_path.exists():
        logging.error(f"Dataset directory {dataset_dir_path} does not exist.")
        return []
    
    data_root = dataset_dir_path / "data"
    if not data_root.exists():
        logging.error(f"Data root {data_root} does not exist.")
        return []
    
    chunk_folders = [f for f in data_root.iterdir() if f.is_dir() and f.name.startswith("chunk-")]
    if not chunk_folders:
        logging.warning(f"No chunk folders found in {data_root}.") # Warning instead of error
        # Check if parquets are directly in data_root (less common for OXE)
        parquet_files_in_data_root = [str(f) for f in data_root.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        if parquet_files_in_data_root:
            logging.info(f"Found {len(parquet_files_in_data_root)} parquet files directly in {data_root}.")
            parquet_paths.extend(parquet_files_in_data_root)
        else:
            return [] # No parquets found

    for chunk_folder in chunk_folders:
        parquet_files = [str(f) for f in chunk_folder.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        if not parquet_files:
            logging.warning(f"No parquet files found in {chunk_folder}.")
            continue
        parquet_paths.extend(parquet_files)
    
    logging.info(f"Found {len(parquet_paths)} parquet files in {dataset_dir_path}.")
    return parquet_paths


### --- process dataset (manages ThreadPool for its parquets) --- ###
def process_dataset(dataset_path_str: str, args):
    dataset_path = Path(dataset_path_str)
    output_dataset_path = dataset_path

    if not global_state.OVERWRITE_ORIGINAL_DATASET:
        if not args.output_home:
            logging.error("Output home directory is not specified (--output-home) and inplace is not 1. Exiting.")
            # This case should ideally not proceed for this dataset.
            # Add to failed list or handle appropriately.
            with global_state.failed_dataset_list_lock:
                global_state.failed_dataset_list.append({
                    "path": dataset_path_str,
                    "reason": "Output home not specified and not inplace."
                })
            return

        output_dataset_path = Path(args.output_home) / dataset_path.name
        if output_dataset_path.exists() and output_dataset_path.is_dir():
            logging.info(f"Output dataset path {output_dataset_path} already exists. Removing it...")
            shutil.rmtree(output_dataset_path)
        elif output_dataset_path.exists() and not output_dataset_path.is_dir():
            logging.error(f"Output path {output_dataset_path} exists and is not a directory. Please remove it manually.")
            with global_state.failed_dataset_list_lock:
                global_state.failed_dataset_list.append({
                    "path": dataset_path_str,
                    "reason": f"Output path {output_dataset_path} conflict."
                })
            return

        logging.info(f"Copying dataset {dataset_path.name} to {output_dataset_path}...")
        try:
            shutil.copytree(dataset_path, output_dataset_path, dirs_exist_ok=True) # dirs_exist_ok for resilience
            logging.info(f"Dataset {dataset_path.name} copied to {output_dataset_path}")
        except Exception as e:
            logging.error(f"Failed to copy {dataset_path.name} to {output_dataset_path}: {e}")
            with global_state.failed_dataset_list_lock:
                global_state.failed_dataset_list.append({
                    "path": dataset_path_str,
                    "reason": f"Failed to copy to output: {e}"
                })
            return
    
    # Load tasks.jsonl for this dataset
    task_json_path = output_dataset_path / "meta" / "tasks.jsonl"
    tasks_df = None
    tasks_df_json_str = None
    if USE_SUBTASK_CONDITIONING_GLOBAL: # Only load if global flag is true
        if task_json_path.exists():
            try:
                tasks_df = pandas.read_json(task_json_path, lines=True)
                tasks_df_json_str = tasks_df.to_json(orient="records", lines=True)
            except Exception as e:
                logging.warning(f"Failed to load or serialize tasks.jsonl from {task_json_path}: {e}. Subtask conditioning might be affected for this dataset.")
        else:
            logging.warning(f"tasks.jsonl not found at {task_json_path}. Subtask conditioning disabled for {dataset_path.name}.")

    all_parquet_file_paths_str = load_parquet_paths_from_dataset(output_dataset_path)
    if not all_parquet_file_paths_str:
        logging.warning(f"No parquet files found to process in {output_dataset_path}. Skipping dataset.")
        return # Nothing to do for this dataset

    parquets_to_process_jobs = []
    config_snapshot = (USE_SUBTASK_CONDITIONING_GLOBAL, SHOW_OBJECT_NAME, SAVE_INSPECTION_IMAGE, RANDOM_SAMPLE_RATE)

    for pf_path_str in all_parquet_file_paths_str:
        if global_state.OVERWRITE_ORIGINAL_DATASET and global_state.finished_parquet_list is not None:
            with global_state.finished_parquet_list_lock: # Ensure thread-safe read
                if pf_path_str in global_state.finished_parquet_list:
                    logging.info(f"Skipping already processed parquet (inplace): {pf_path_str}")
                    continue
        parquets_to_process_jobs.append(
            (pf_path_str, tasks_df_json_str, str(output_dataset_path)) + config_snapshot
        )

    if not parquets_to_process_jobs:
        logging.info(f"No new parquet files to process in dataset {dataset_path.name}.")
        # If all were skipped, it's a success for this dataset in terms of not needing work
        # However, bboxes.jsonl might still need to be generated if it's missing and parquets exist
        # For now, assume if no new jobs, this dataset is fine.
        return

    all_worker_results = []
    dataset_failed_a_parquet = False
    first_failed_parquet_schema_dict = None # For reporting in failed_dataset_list

    # Initialize ThreadPoolExecutor here, for this dataset's parquets
    # The initializer init_worker_model_on_thread has already been set up by main for the global pool
    # We are reusing the global executor created in main.
    # This means process_dataset itself is not creating the executor.
    # It should receive the executor as an argument or use a global one.
    # For simplicity, using a global executor defined in main.

    # The executor is passed from main or created here if process_dataset is meant to be standalone later
    # Assuming executor is accessible (e.g. passed or global from main's ThreadPoolExecutor context)
    # The current structure uses one executor created in main.
    
    # The tqdm progress bar for parquets within this dataset
    results_iterator = args.executor.map(process_parquet_task, parquets_to_process_jobs)
    for result in tqdm(results_iterator, total=len(parquets_to_process_jobs), desc=f"Parquets in {dataset_path.name}"):
        all_worker_results.append(result)
        
        # Unpack result: (processed_parquet_path, schema_dict, records, r_bboxes, error_flag, error_msg)
        _, schema_dict, _, _, error_flag, _ = result
        if error_flag:
            dataset_failed_a_parquet = True
            if first_failed_parquet_schema_dict is None: # Capture schema from the first failure
                first_failed_parquet_schema_dict = schema_dict 

    if dataset_failed_a_parquet:
        logging.error(f"One or more parquets failed processing in dataset {dataset_path.name}. Dataset processing aborted.")
        with global_state.failed_dataset_list_lock:
            global_state.failed_dataset_list.append({
                "path": dataset_path_str,
                "reason": "Failure in processing one or more parquet files.",
                "features_schema_of_a_failed_parquet": first_failed_parquet_schema_dict # Can be None if load failed early
            })
        return

    # If all parquets processed successfully by workers (image processing stage complete)
    # Now, Stage 2: Aggregate, assign bbox_indices, and save parquets and bboxes.jsonl
    # This part is sequential for data integrity of bbox_index and bboxes.jsonl.
    
    current_bbox_global_idx = 0
    dataset_final_bboxes_for_jsonl = []
    
    logging.info(f"Aggregating results and saving parquets for dataset {dataset_path.name}...")
    for res_parquet_path_str, res_schema_dict, res_records, res_r_bboxes_list, _, _ in all_worker_results:
        if not res_records and not res_r_bboxes_list: # Should be caught by error_flag, but double check
            logging.warning(f"No records returned for {res_parquet_path_str}, possibly an error not flagged. Skipping save for this file.")
            continue

        # Reconstruct features for saving
        if res_schema_dict is None:
            logging.error(f"Missing schema for {res_parquet_path_str}. Cannot save.")
            # Mark dataset as failed if this happens
            dataset_failed_a_parquet = True # set flag again
            break 
        
        original_features = Features.from_dict(res_schema_dict)
        new_features = original_features.copy()
        new_features['bbox_index'] = Value('int64')

        records_to_save_with_bbox_idx = []
        for record_idx_in_file, (processed_record, r_bbox_item) in enumerate(zip(res_records, res_r_bboxes_list)):
            # Ensure record is a dict for modification
            record_to_save = dict(processed_record)
            record_to_save['bbox_index'] = current_bbox_global_idx
            records_to_save_with_bbox_idx.append(record_to_save)
            
            dataset_final_bboxes_for_jsonl.append({
                'bbox_index': current_bbox_global_idx,
                'bbox': r_bbox_item
            })
            current_bbox_global_idx += 1
        
        # Save the updated parquet file
        try:
            if records_to_save_with_bbox_idx: # Ensure there's data to save
                updated_dataset = Dataset.from_list(records_to_save_with_bbox_idx, features=new_features)
            else: # Create an empty dataset with the new schema if no records (e.g. empty input parquet)
                updated_dataset = Dataset.from_list([], features=new_features)
            
            updated_dataset.to_parquet(str(res_parquet_path_str))
            print(f"Processed parquet {str(res_parquet_path_str)} successfully") # Original print

            if global_state.OVERWRITE_ORIGINAL_DATASET:
                with global_state.finished_parquet_list_lock:
                    if global_state.finished_parquet_list is not None: # Check again, might be set to None by race
                         global_state.finished_parquet_list.append(str(res_parquet_path_str))
        except Exception as e:
            logging.error(f"Error saving updated dataset to {res_parquet_path_str}: {e}", exc_info=True)
            print(f"Error saving updated dataset to {res_parquet_path_str}: {e}") # Original print
            dataset_failed_a_parquet = True # Mark failure for the dataset
            break # Stop processing this dataset further if a save fails

    if dataset_failed_a_parquet: # Check if any failure occurred during save stage
        logging.error(f"Failed to save one or more processed parquets for dataset {dataset_path.name}. Dataset processing marked as failed.")
        with global_state.failed_dataset_list_lock:
            # Check if already added, if so, update reason or just log
            existing_failure = next((item for item in global_state.failed_dataset_list if item["path"] == dataset_path_str), None)
            if not existing_failure:
                global_state.failed_dataset_list.append({
                    "path": dataset_path_str,
                    "reason": "Failure in saving one or more parquet files after processing.",
                    "features_schema_of_a_failed_parquet": first_failed_parquet_schema_dict # From earlier stage
                })
        return

    # Save bboxes.jsonl for the dataset
    bboxes_json_path = output_dataset_path / "meta" / "bboxes.jsonl"
    try:
        df_bboxes = pandas.DataFrame(dataset_final_bboxes_for_jsonl)
        df_bboxes.to_json(bboxes_json_path, orient="records", lines=True, force_ascii=False)
        print(f"Processed dataset {dataset_path_str} successfully. Bboxes saved to {bboxes_json_path}.") # Original print
    except Exception as e:
        logging.error(f"Failed to save bboxes.jsonl for {dataset_path.name} to {bboxes_json_path}: {e}", exc_info=True)
        with global_state.failed_dataset_list_lock:
             global_state.failed_dataset_list.append({
                "path": dataset_path_str,
                "reason": f"Failed to save bboxes.jsonl: {e}"
            })


### --- main function --- ###
def main_process(args):
    global executor_ref

    # Setup based on args
    global_state.OVERWRITE_ORIGINAL_DATASET = (args.inplace == 1)
    if global_state.OVERWRITE_ORIGINAL_DATASET:
        if not args.finished_list_path:
            logging.error("When --inplace=1, --finished-list-path must be provided. Exiting.")
            sys.exit(1)
        global_state.args_finished_list_path = args.finished_list_path
        try:
            if Path(args.finished_list_path).exists():
                with open(args.finished_list_path, "r", encoding="utf-8") as f:
                    global_state.finished_parquet_list = json.load(f)
                logging.info(f"Loaded {len(global_state.finished_parquet_list)} paths from {args.finished_list_path}")
            else:
                global_state.finished_parquet_list = []
                logging.info(f"Finished list file {args.finished_list_path} not found. Starting with an empty list.")
        except json.JSONDecodeError:
            logging.error(f"Could not decode JSON from {args.finished_list_path}. Starting with an empty list.")
            global_state.finished_parquet_list = [] # Fallback to empty list

        # Setup exit handlers for saving finished_parquet_list
        # def save_finished_list_on_exit(g_state_ref):
        #     if g_state_ref.OVERWRITE_ORIGINAL_DATASET and g_state_ref.finished_parquet_list is not None:
        #         logging.info(f"Saving finished_parquet_list to {g_state_ref.args_finished_list_path} before exit...")
        #         try:
        #             # Use lock to prevent race condition if another thread is modifying it during shutdown
        #             with g_state_ref.finished_parquet_list_lock:
        #                 with open(g_state_ref.args_finished_list_path, "w", encoding="utf-8") as f:
        #                     json.dump(g_state_ref.finished_parquet_list, f, ensure_ascii=False, indent=2)
        #         except Exception as e:
        #             logging.error(f"Failed to save finished_parquet_list: {e}")
        def save_finished_list_on_exit(g_state_ref):
            if g_state_ref.OVERWRITE_ORIGINAL_DATASET and g_state_ref.finished_parquet_list is not None:
                logging.info(f"Attempting to save finished_parquet_list to {g_state_ref.args_finished_list_path}...")
                # Try to acquire the lock without blocking
                lock_acquired = g_state_ref.finished_parquet_list_lock.acquire(blocking=False)      
                if lock_acquired:
                    try:
                        with open(g_state_ref.args_finished_list_path, "w", encoding="utf-8") as f:
                            json.dump(g_state_ref.finished_parquet_list, f, ensure_ascii=False, indent=2)
                        logging.info("Successfully saved finished_parquet_list.")
                    except Exception as e:
                        logging.error(f"Failed to save finished_parquet_list: {e}")
                    finally:
                        g_state_ref.finished_parquet_list_lock.release() # Crucial to release the lock
                else:
                    logging.warning("Could not acquire lock for finished_parquet_list immediately. Skipping save on exit to prevent deadlock.")

        atexit.register(save_finished_list_on_exit, global_state)
        
        def handle_sigint_custom(signum, frame, g_state_ref):
            global executor_ref
            logging.warning("Caught Ctrl+C (SIGINT), attempting to save finished_parquet_list...")
            if executor_ref:
                logging.warning("Shutting down ThreadPoolExecutor (no new tasks, wait for running)...")
                executor_ref.shutdown(wait=False)
            save_finished_list_on_exit(g_state_ref)
            sys.exit(1) # Exit after attempting to save
        signal.signal(signal.SIGINT, lambda s, f: handle_sigint_custom(s, f, global_state))

    os.makedirs("inspection_images", exist_ok=True)

    # Determine available CUDA devices FOR PYTORCH
    if not torch.cuda.is_available():
        logging.error("CUDA is not available according to PyTorch! Exiting.")
        sys.exit(1)

    num_pytorch_devices = torch.cuda.device_count()
    if num_pytorch_devices == 0:
        # This can happen if CUDA_VISIBLE_DEVICES is set to empty or invalid devices not recognized by the driver/pytorch
        logging.error(f"PyTorch reports CUDA is available, but torch.cuda.device_count() is 0. "
                      f"Check your CUDA_VISIBLE_DEVICES setting. Currently: '{os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}'")
        sys.exit(1)
    
    # global_state.pytorch_device_indices will store PyTorch relative indices [0, 1, ..., N-1]
    global_state.pytorch_device_indices = list(range(num_pytorch_devices))
    
    # Log information about device mapping
    cuda_visible_devices_env_var = os.environ.get('CUDA_VISIBLE_DEVICES', 'All (Not Set)')
    logging.info(f"CUDA_VISIBLE_DEVICES is set to: '{cuda_visible_devices_env_var}'.")
    logging.info(f"PyTorch sees {num_pytorch_devices} CUDA device(s).")
    logging.info(f"Workers will be assigned PyTorch device indices: {global_state.pytorch_device_indices} in a round-robin fashion.")
    if cuda_visible_devices_env_var != 'All (Not Set)':
        physical_ids_parsed = [d.strip() for d in cuda_visible_devices_env_var.split(',') if d.strip()]
        if len(physical_ids_parsed) == num_pytorch_devices:
            mapping_info = ", ".join([f"PyTorch cuda:{i} -> Physical ID {physical_ids_parsed[i]}" for i in range(num_pytorch_devices)])
            logging.info(f"Mapping: {mapping_info}")
        else:
            logging.warning("Mismatch between parsed CUDA_VISIBLE_DEVICES and torch.cuda.device_count(). Manual verification of mapping might be needed.")

    # Create a single ThreadPoolExecutor for the entire run
    # The initializer will be called for each of the num_proc workers once.
    with ThreadPoolExecutor(max_workers=args.num_proc, initializer=init_worker_model_on_thread) as executor:
        executor_ref = executor 
        args.executor = executor # Make executor available to process_dataset via args

        if args.dataset_path:
            logging.info(f"Processing single dataset: {args.dataset_path}")
            process_dataset(args.dataset_path, args)
        elif args.dataset_home:
            dataset_home_path = Path(args.dataset_home)
            all_datasets_paths = [str(f) for f in dataset_home_path.iterdir() if f.is_dir()]
            logging.info(f"Found {len(all_datasets_paths)} potential datasets in {args.dataset_home}.")
            for i, ds_path_str in enumerate(all_datasets_paths):
                logging.info(f"--- Processing dataset {i+1}/{len(all_datasets_paths)}: {ds_path_str} ---")
                process_dataset(ds_path_str, args)
        else:
            logging.error("Please provide either --dataset-path or --dataset-home.")
            sys.exit(1)
    executor_ref = None

    # Save failed datasets if any, after executor is shut down
    if global_state.failed_dataset_list:
        logging.info(f"There are {len(global_state.failed_dataset_list)} datasets that failed or had errors. Details saved to './failed_datasets.jsonl'.")
        failed_datasets_path = Path("failed_datasets.jsonl")
        try:
            df_failed = pandas.DataFrame(global_state.failed_dataset_list)
            df_failed.to_json(failed_datasets_path, orient="records", lines=True, force_ascii=False)
        except Exception as e:
            logging.error(f"Could not save failed_datasets.jsonl: {e}")
            logging.error(f"Failed datasets data: {json.dumps(global_state.failed_dataset_list, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate through and process LeRobot datasets with grounding.")
    parser.add_argument("--dataset-path", type=str, help="Path of a single LeRobot dataset.")
    parser.add_argument("--dataset-home", type=str, help="Home directory of multiple LeRobot datasets.")
    parser.add_argument("--output-home", type=str, help="Output home directory for processed data (if not inplace).")
    parser.add_argument("--inplace", type=int, default=0, choices=[0, 1],
                        help="If 1, overwrite original dataset. Requires --finished-list-path.")
    parser.add_argument("--finished-list-path", type=str,
                        help="Path to a JSON file storing list of already processed parquet files (for inplace mode).")
    parser.add_argument("--num-proc", type=int, default=4, help="Number of worker processes (threads).")
    
    parsed_args = parser.parse_args()

    # TOKENIZERS_PARALLELISM should be false for threads with transformers
    if os.environ.get("TOKENIZERS_PARALLELISM", "").lower() == "true":
        logging.warning("TOKENIZERS_PARALLELISM is 'true'. For multithreading with Hugging Face transformers, it's often recommended to set it to 'false' (e.g., `export TOKENIZERS_PARALLELISM=false`) to avoid potential deadlocks. The script will continue, but consider this if issues arise.")
    elif "TOKENIZERS_PARALLELISM" not in os.environ:
        logging.info("TOKENIZERS_PARALLELISM is not set. Setting to 'false' for this script run to be safe with multithreading.")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    main_process(parsed_args)