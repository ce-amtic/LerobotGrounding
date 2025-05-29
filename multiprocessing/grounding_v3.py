"""
CUDA_VISIBLE_DEVICES=6,7 TOKENIZERS_PARALLELISM=false python grounding_v4.py \
    --dataset-path /pdata/oxe_lerobot/dlr_sara_pour_converted_externally_to_rlds \
    --output-home /pdata/oxe_lerobot_g \
    --num-proc 8 > o.txt

CUDA_VISIBLE_DEVICES=6,7 TOKENIZERS_PARALLELISM=false python grounding_v4.py \
    --dataset-home /pdata/oxe_lerobot \
    --inplace 1 \
    --num-proc 8 > o.txt
"""

import os
import sys
import io
import argparse
import logging
import pandas # Keep for tasks.jsonl and bboxes.jsonl and failed_dataset_list
import shutil
import random
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Dataset, Value, Features
from datasets.features.image import Image as ImageHF
import multiprocessing as mp

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if os.getenv("DEBUG"):
    logging.getLogger().setLevel(logging.DEBUG)

# --- Global Configuration (from original script, will be passed or used carefully) ---
# These are now more like default configs, actual behavior might be influenced by args
# For example, OVERWRITE_ORIGINAL_DATASET is set based on --inplace arg.
SHOW_OBJECT_NAME_CONFIG = False
USE_SUBTASK_CONDITIONING_CONFIG = True
SAVE_INSPECTION_IMAGE_CONFIG = True
RANDOM_SAMPLE_RATE_CONFIG = 0.0005
MODEL_NAME_CONFIG = "/fyh/.env/models/Qwen2.5-VL-7B-Instruct"
# This will be set in main based on cli_args.inplace
OVERWRITE_ORIGINAL_DATASET_BEHAVIOR = False


# --- Global State for Worker Processes (set by initializer) ---
worker_q_model = None
worker_q_processor = None

# --- Other Global State (managed by main process or carefully in parallel contexts) ---
# inspection_image_id needs to be a global counter if images are saved from main process
# after collecting results from workers.
inspection_image_id_counter = 0
os.makedirs("inspection_images", exist_ok=True) # Create once

# This list will be populated by the main processing logic if datasets/files fail
# It's appended to by the main thread, so it's safe.
main_failed_dataset_list = []

IMAGE_TYPE_HF_ENUM = 1
IMAGE_TYPE_DICT_BYTES_ENUM = 2


# --- Initializer for multiprocessing.Pool workers ---
def init_worker_process_for_pool():
    global worker_q_model, worker_q_processor
    from qwen_grounding_utils import setup_model # Import here for worker process

    pid = os.getpid()
    if worker_q_model is None or worker_q_processor is None: # Ensure it's loaded only once per worker
        logging.info(f"Worker {pid}: Initializing Qwen model ({MODEL_NAME_CONFIG})...")
        try:
            worker_q_model, worker_q_processor = setup_model(MODEL_NAME_CONFIG)
            logging.info(f"Worker {pid}: Qwen model initialized and ready.")
        except Exception as e_setup:
            logging.error(f"Worker {pid}: CRITICAL - Failed to setup model: {e_setup}")
            raise RuntimeError(f"Worker {pid} failed to initialize model") from e_setup
    # else: logging.debug(f"Worker {pid}: Model already initialized.")


# --- Worker function for the Pool ---
def _apply_grounding_to_record_task_for_pool(task_args_dict):
    global worker_q_model, worker_q_processor
    from qwen_grounding_utils import grounding_pipeline

    if worker_q_model is None or worker_q_processor is None:
        return {"error": "Model not loaded in worker", "original_record_dict": task_args_dict.get("record_dict")}

    record_dict = task_args_dict["record_dict"]
    # Unpack other args from task_args_dict similar to the previous version...
    tasks_df = task_args_dict["tasks_df"]
    image_cols_cfg = task_args_dict["image_cols_cfg"]
    task_idx_col_exists = task_args_dict["task_idx_col_exists"]
    parquet_path = task_args_dict["parquet_path"]
    # Configs
    show_obj_name_cfg = task_args_dict["show_obj_name_cfg"]
    use_subtask_cfg = task_args_dict["use_subtask_cfg"]
    save_insp_img_cfg = task_args_dict["save_insp_img_cfg"]
    rand_sample_cfg = task_args_dict["rand_sample_cfg"]

    processed_record = dict(record_dict)
    r_task_index_val = processed_record.get('task_index', None)
    task_desc_str = None
    can_use_task = False

    if use_subtask_cfg and tasks_df is not None and task_idx_col_exists:
        if r_task_index_val is not None:
            try:
                task_row = tasks_df[tasks_df['task_index'] == r_task_index_val]
                if not task_row.empty:
                    task_desc_str = task_row.iloc[0]['task']
                    can_use_task = True
            except Exception: pass
    final_use_task_pipeline = use_subtask_cfg and can_use_task

    record_bbox_json = None
    inspection_info = None

    for img_col_name, img_type_enum in image_cols_cfg:
        img_data = processed_record.get(img_col_name)
        pil_img_in = None
        try:
            if img_type_enum == IMAGE_TYPE_HF_ENUM:
                if isinstance(img_data, Image.Image): pil_img_in = img_data
                elif isinstance(img_data, dict) and 'bytes' in img_data and isinstance(img_data['bytes'], bytes):
                    pil_img_in = Image.open(io.BytesIO(img_data['bytes']))
            elif img_type_enum == IMAGE_TYPE_DICT_BYTES_ENUM:
                if isinstance(img_data, dict) and 'bytes' in img_data and isinstance(img_data['bytes'], bytes):
                    pil_img_in = Image.open(io.BytesIO(img_data['bytes']))
        except Exception: pil_img_in = None

        if pil_img_in is None:
            if record_bbox_json is None: record_bbox_json = {"error": f"No image for {img_col_name}"}
            continue

        json_resp, out_img = grounding_pipeline(
            image=pil_img_in, model=worker_q_model, processor=worker_q_processor,
            task_desc=task_desc_str if final_use_task_pipeline else None,
            SHOW_OBJECT_NAME=show_obj_name_cfg, USE_SUBTASK_CONDITIONING=final_use_task_pipeline
        )

        if save_insp_img_cfg and out_img and (random.random() < rand_sample_cfg):
            if inspection_info is None:
                buffer = io.BytesIO(); fmt_i = out_img.format or "JPEG"
                out_img.save(buffer, format=fmt_i)
                inspection_info = {
                    'image_bytes': buffer.getvalue(), 'image_format': fmt_i,
                    'parquet_path': parquet_path, 'task_desc': task_desc_str if final_use_task_pipeline else '',
                    'json_response': json_resp,
                }
        
        if out_img is not None:
            if img_type_enum == IMAGE_TYPE_HF_ENUM:
                # Convert to bytes for pickling, main process reconstructs
                img_b_buf = io.BytesIO(); img_fmt = out_img.format or "PNG"
                out_img.save(img_b_buf, format=img_fmt)
                processed_record[img_col_name] = {'_pil_bytes_': img_b_buf.getvalue(), '_pil_format_': img_fmt}
            elif img_type_enum == IMAGE_TYPE_DICT_BYTES_ENUM:
                buf = io.BytesIO(); fmt_o = out_img.format or "PNG"
                out_img.save(buf, format=fmt_o)
                if isinstance(processed_record.get(img_col_name), dict): processed_record[img_col_name]['bytes'] = buf.getvalue()
                else: processed_record[img_col_name] = {'bytes': buf.getvalue()}
        
        if record_bbox_json is None: record_bbox_json = json_resp
    
    return {
        "processed_record_content": processed_record,
        "bbox_json_response": record_bbox_json,
        "inspection_info": inspection_info
    }


def _get_image_columns_config_from_features(features_obj: Features):
    image_columns = []
    # Prioritize "observation.images.cam" as per original logic
    # Make sure to iterate over actual column names present in features_obj
    all_cols_in_features = list(features_obj.keys()) # Get all column names

    suspected_cam_cols = [col for col in all_cols_in_features if col.startswith("observation.images.cam")]
    other_cols = [col for col in all_cols_in_features if not col.startswith("observation.images.cam")]

    for col_name in suspected_cam_cols + other_cols: # Check suspected first
        if col_name not in features_obj: # Should not happen if all_cols_in_features is from features_obj.keys()
            continue
            
        feature_type = features_obj[col_name] # This is the FeatureType object for the column

        if isinstance(feature_type, ImageHF):
            image_columns.append((col_name, IMAGE_TYPE_HF_ENUM))
            logging.debug(f"Identified ImageHF column: {col_name}")
        elif isinstance(feature_type, dict) and 'bytes' in feature_type:
            # Here, feature_type is a dictionary, e.g., {'bytes': Value(dtype='binary', ...)}
            # The actual type of the 'bytes' field is feature_type['bytes']
            bytes_field_type = feature_type['bytes']
            is_binary_value = False

            if isinstance(bytes_field_type, Value):
                if bytes_field_type.dtype == 'binary':
                    is_binary_value = True
            
            if is_binary_value:
                if col_name.startswith("observation.images.cam"):
                    image_columns.append((col_name, IMAGE_TYPE_DICT_BYTES_ENUM))
                    logging.debug(f"Identified DICT_BYTES (Value binary) column: {col_name}")
            
    if not image_columns:
        logging.warning(f"No image columns identified with current heuristics from features: {features_obj}")
    return image_columns


def _process_parquet_file_with_pool_final(process_pool_instance: mp.Pool,
                                          parquet_file_path: str,
                                          tasks_dataframe: pandas.DataFrame = None):
    """
    Processes a single Parquet file using the provided persistent multiprocessing pool.
    This function is called by the main dataset processing loop.
    Returns tuple: (list_of_processed_records, list_of_bbox_data, list_of_inspection_info, original_features_obj)
    or (None, None, None, None) on critical failure.
    """
    global main_failed_dataset_list # Use the main process's list for logging file-level errors

    try:
        episode_dataset = load_dataset("parquet", data_files=str(parquet_file_path), split='train', keep_in_memory=True)
        original_features = episode_dataset.features
    except Exception as e:
        logging.error(f"PoolWrapper: Error loading {parquet_file_path}: {e}")
        # Do not append to main_failed_dataset_list here, as that's for *dataset* level failures.
        # This function's caller (process_dataset_final) will handle if this file contributes to dataset failure.
        return None, None, None, None # Signal failure for this file

    image_cols_config = _get_image_columns_config_from_features(original_features)
    if not image_cols_config:
        logging.warning(f"PoolWrapper: No image columns found for {parquet_file_path}. This file will not be grounded.")
        # Return original records to be saved as-is, without bbox_index.
        # The main loop will need to handle this by not adding bbox_index or adding a placeholder.
        # For faithful reproduction, if original process_parquet returned False, features,
        # this means the dataset processing should record a failure for this parquet file step.
        # Let's return the records and original features, indicating no grounding was done.
        # The caller can then decide if this constitutes a dataset failure.
        # We will return empty lists for bboxes and inspection for this file.
        return list(episode_dataset), [], [], original_features


    task_index_col_exists = 'task_index' in original_features

    # Prepare list of task arguments for the pool
    tasks_to_submit_to_pool = []
    for record_data_dict in episode_dataset:
        task_arg_dict = {
            "record_dict": record_data_dict,
            "tasks_df": tasks_dataframe, # Renamed for clarity from tasks_df_for_map
            "image_cols_cfg": image_cols_config,
            "task_idx_col_exists": task_index_col_exists,
            "parquet_path": parquet_file_path,
            "show_obj_name_cfg": SHOW_OBJECT_NAME_CONFIG,
            "use_subtask_cfg": USE_SUBTASK_CONDITIONING_CONFIG,
            "save_insp_img_cfg": SAVE_INSPECTION_IMAGE_CONFIG,
            "rand_sample_cfg": RANDOM_SAMPLE_RATE_CONFIG,
        }
        tasks_to_submit_to_pool.append(task_arg_dict)

    if not tasks_to_submit_to_pool: # Empty Parquet file
        return [], [], [], original_features

    # Submit tasks to the pool
    # Using map for simplicity here, assumes order of results matches input order of tasks
    # tqdm can be used with imap_unordered if preferred for progress on long tasks
    # chunksize = max(1, len(tasks_to_submit_to_pool) // (process_pool_instance._processes * 2 if hasattr(process_pool_instance, '_processes') else 2))
    try:
        # Using a simple map which blocks. tqdm for the outer file loop is probably sufficient.
        # For more granular progress, use imap_unordered with tqdm.
        pool_results_list = process_pool_instance.map(_apply_grounding_to_record_task_for_pool, tasks_to_submit_to_pool) #, chunksize=chunksize)
    except Exception as e_pool_exec:
        logging.error(f"PoolWrapper: Critical error during pool execution for {parquet_file_path}: {e_pool_exec}")
        # This likely means a worker died or a fundamental issue with tasks.
        return None, None, None, None # Signal critical failure for this file


    # Process results from the pool
    file_processed_records = []
    file_bboxes_data = []
    file_inspection_infos = []

    for result_payload_dict in pool_results_list:
        if isinstance(result_payload_dict, dict) and "error" in result_payload_dict:
            logging.warning(f"PoolWrapper: Worker returned error for a record in {parquet_file_path}: {result_payload_dict['error']}")
            # If worker returns an error, we might want to keep the original record content
            # The original script's process_parquet didn't explicitly handle per-record failures,
            # it either completed a file or the whole file processing was part of a larger failure.
            # Let's append the (possibly unmodified or partially modified) record content.
            record_content_on_error = result_payload_dict.get("processed_record_content", result_payload_dict.get("original_record_dict", {}))
            file_processed_records.append(record_content_on_error)
            file_bboxes_data.append(None) # No bbox if error
            continue

        record_content = result_payload_dict["processed_record_content"]
        # Reconstruct PIL images if they were returned as bytes
        for img_col, _ in image_cols_config: # Use the config to know which cols were images
            if isinstance(record_content.get(img_col), dict) and '_pil_bytes_' in record_content[img_col]:
                try:
                    pil_info = record_content[img_col]
                    record_content[img_col] = Image.open(io.BytesIO(pil_info['_pil_bytes_']))
                except Exception as e_reconstruct:
                    logging.warning(f"PoolWrapper: Failed to reconstruct PIL for {img_col} from {parquet_file_path}: {e_reconstruct}")
                    # record_content[img_col] remains the dict with bytes, or could be set to None
        
        file_processed_records.append(record_content)
        file_bboxes_data.append(result_payload_dict["bbox_json_response"])
        if result_payload_dict["inspection_info"]:
            file_inspection_infos.append(result_payload_dict["inspection_info"])
            
    return file_processed_records, file_bboxes_data, file_inspection_infos, original_features


# --- Functions to replicate original script's top-level logic ---
def _load_parquet_files_from_dataset_path(dataset_p: Path):
    # This is the `load_parquet` function from the original script
    parquet_paths = []
    if not dataset_p.exists():
        logging.error(f"Dataset in {dataset_p} does not exist.")
        # Original script did sys.exit(1). Here, we might want to let the caller handle it.
        return parquet_paths # Empty list

    data_root = dataset_p / "data"
    if not data_root.is_dir(): # Check if it's a directory
        logging.error(f"Data root {data_root} does not exist or not a directory.")
        return parquet_paths

    chunk_folders = [f for f in data_root.iterdir() if f.is_dir() and f.name.startswith("chunk-")]
    if not chunk_folders:
        # Original script did sys.exit(1). For robustness, log and return empty.
        logging.error(f"No chunk folders found in {data_root}.")
        # Check if parquet files exist directly in data_root as a fallback
        direct_parquets_in_data = [str(f) for f in data_root.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        if direct_parquets_in_data:
            logging.info(f"Found {len(direct_parquets_in_data)} parquet files directly in {data_root}.")
            return direct_parquets_in_data
        return parquet_paths

    for chunk_folder in chunk_folders:
        parquet_files_in_chunk = [str(f) for f in chunk_folder.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        if not parquet_files_in_chunk:
            logging.warning(f"No parquet files found in {chunk_folder}.") # Original logged error and continued
            continue
        parquet_paths.extend(parquet_files_in_chunk)
    
    logging.info(f"Found {len(parquet_paths)} parquet files in {dataset_p}.")
    return parquet_paths


def process_single_dataset_final(dataset_path_str: str, cli_args, process_pool_instance: mp.Pool):
    """
    Processes a single dataset. This function mirrors the logic of the original
    `process_dataset` function but uses the parallel parquet processing.
    """
    global inspection_image_id_counter, main_failed_dataset_list, OVERWRITE_ORIGINAL_DATASET_BEHAVIOR

    current_dataset_path_obj = Path(dataset_path_str)
    dataset_succeeded = True # Flag for this dataset

    # Determine effective path (original or copy)
    if OVERWRITE_ORIGINAL_DATASET_BEHAVIOR:
        effective_dataset_path = current_dataset_path_obj
        logging.info(f"Processing dataset IN-PLACE: {effective_dataset_path}")
    else:
        if not cli_args.output_home:
            logging.error("Output home directory is not specified and not in-place. Cannot proceed.")
            main_failed_dataset_list.append({"path": dataset_path_str, "reason": "No output_home for non-inplace."})
            return # Skip this dataset
        effective_dataset_path = Path(cli_args.output_home) / current_dataset_path_obj.name
        logging.info(f"Output for {current_dataset_path_obj.name} will be in: {effective_dataset_path}")
        try:
            if effective_dataset_path.exists():
                logging.info(f"Removing existing output path: {effective_dataset_path}")
                shutil.rmtree(effective_dataset_path)
            shutil.copytree(current_dataset_path_obj, effective_dataset_path, dirs_exist_ok=False)
            logging.info(f"Copied {current_dataset_path_obj.name} to {effective_dataset_path}")
        except Exception as e_copy:
            logging.error(f"Failed to copy {current_dataset_path_obj.name} to {effective_dataset_path}: {e_copy}")
            main_failed_dataset_list.append({"path": dataset_path_str, "reason": f"Copy failed: {e_copy}"})
            return

    # Load tasks.jsonl for this dataset
    tasks_df_for_dataset = None
    task_json_path = effective_dataset_path / "meta" / "tasks.jsonl"
    if task_json_path.exists():
        try:
            tasks_df_for_dataset = pandas.read_json(task_json_path, lines=True)
        except Exception as e_load_tasks:
            logging.error(f"Failed to load tasks from {task_json_path}: {e_load_tasks}")
            if USE_SUBTASK_CONDITIONING_CONFIG:
                logging.warning("Subtask conditioning may be affected.")
    elif USE_SUBTASK_CONDITIONING_CONFIG:
        logging.warning(f"tasks.jsonl not found at {task_json_path}. Subtask conditioning will be affected.")

    # Get list of Parquet files for this dataset
    parquet_files_to_process = _load_parquet_files_from_dataset_path(effective_dataset_path)
    if not parquet_files_to_process:
        logging.error(f"No Parquet files found to process in {effective_dataset_path}.")
        # Original script's process_parquet would effectively lead to a failure for the dataset.
        # Let's record this as a dataset failure.
        main_failed_dataset_list.append({
            "path": dataset_path_str, # Original path for user reference
            "effective_path": str(effective_dataset_path),
            "reason": "No Parquet files found in dataset.",
            "features_if_any": None # Mimicking original failed_dataset_list structure
        })
        return

    # --- Loop through Parquet files, process them using the pool, and aggregate ---
    dataset_level_all_bboxes = [] # Collect all bboxes for *this* dataset
    
    num_parquet_files_successfully_processed = 0

    for p_file_path_str in tqdm(parquet_files_to_process, desc=f"Parquet files in {current_dataset_path_obj.name}", unit="pfile"):
        # p_file_path_str is the path within the effective_dataset_path (original or copy)
        
        # Process one parquet file using the pool
        # Returns: list_records, list_bboxes_data, list_inspection_data, original_features
        records_list, bboxes_data_list, inspection_data_list, p_file_original_features = \
            _process_parquet_file_with_pool_final(
                process_pool_instance, p_file_path_str, tasks_dataframe=tasks_df_for_dataset
            )

        if records_list is None: # Critical failure in processing this Parquet file
            logging.error(f"Critical failure processing {p_file_path_str}. It will be skipped for saving.")
            dataset_succeeded = False # Mark dataset as having issues
            # The error for this file is already logged by _process_parquet_file_with_pool_final
            continue # Move to the next parquet file

        if not p_file_original_features: # Should not happen if records_list is not None
             logging.error(f"Original features not returned for {p_file_path_str} despite records existing. Skipping.")
             dataset_succeeded = False
             continue


        # Post-processing for this single Parquet file's results (in main thread)
        current_p_file_records_with_bbox_idx = []
        if records_list: # If there are records (even if grounding failed for some)
            for i, record_item_dict in enumerate(records_list):
                bbox_data_for_this_record = bboxes_data_list[i] if i < len(bboxes_data_list) else None
                
                # Assign global (dataset-level) bbox_index
                current_dataset_bbox_idx = len(dataset_level_all_bboxes)
                dataset_level_all_bboxes.append({
                    'bbox_index': current_dataset_bbox_idx,
                    'bbox': bbox_data_for_this_record
                })
                
                mut_record = dict(record_item_dict) # Ensure mutable
                mut_record['bbox_index'] = current_dataset_bbox_idx
                current_p_file_records_with_bbox_idx.append(mut_record)

        # Save inspection images (if any) for this Parquet file
        if inspection_data_list and SAVE_INSPECTION_IMAGE_CONFIG:
            for insp_info in inspection_data_list:
                inspection_image_id_counter += 1 # Global counter
                p_path_safe = insp_info['parquet_path'].replace('/', '-').replace('\\', '-').replace('.', '-')
                if insp_info.get('image_bytes'): # Check if image bytes are present
                    try:
                        img_to_save = Image.open(io.BytesIO(insp_info['image_bytes']))
                        img_to_save.save(f"inspection_images/{inspection_image_id_counter}_{p_path_safe}.{insp_info.get('image_format', 'jpg').lower()}")
                        
                        # Save JSON info alongside
                        json_info_to_save = {
                            'original_parquet_path': insp_info['parquet_path'],
                            'task_desc': insp_info['task_desc'],
                            'json_response': insp_info['json_response']
                        }
                        with open(f"inspection_images/{inspection_image_id_counter}_{p_path_safe}.json", 'w') as jf:
                            json.dump(json_info_to_save, jf, ensure_ascii=False, indent=4)
                    except Exception as e_insp:
                        logging.error(f"Error saving inspection image/json for {p_path_safe}: {e_insp}")

        # --- Save the processed Parquet file ---
        # Define features for the new dataset
        final_p_file_features_dict = {name: f_type for name, f_type in p_file_original_features.items()}
        final_p_file_features_dict['bbox_index'] = Value('int64') # Add bbox_index type

        # Ensure image columns have correct feature type after potential PIL->bytes->PIL
        p_file_img_cols_cfg = _get_image_columns_config_from_features(p_file_original_features)
        for img_col, img_type in p_file_img_cols_cfg:
            if img_type == IMAGE_TYPE_HF_ENUM:
                # If records now contain PIL.Image for this column, ImageHF() is correct.
                final_p_file_features_dict[img_col] = ImageHF()
            elif img_type == IMAGE_TYPE_DICT_BYTES_ENUM:
                # This should remain a dict feature.
                final_p_file_features_dict[img_col] = p_file_original_features[img_col]
        
        final_p_file_features_obj = Features(final_p_file_features_dict)

        # Create Dataset from list of records for this Parquet file
        if current_p_file_records_with_bbox_idx:
            dataset_to_save = Dataset.from_list(current_p_file_records_with_bbox_idx, features=final_p_file_features_obj)
        else: # Empty file or all records failed processing inside worker
            dataset_to_save = Dataset.from_list([], features=final_p_file_features_obj)

        try:
            dataset_to_save.to_parquet(p_file_path_str) # Save to the path (original or copied location)
            print(f"Successfully saved processed Parquet: {p_file_path_str}")
            num_parquet_files_successfully_processed +=1
        except Exception as e_save_p:
            logging.error(f"Error saving processed Parquet {p_file_path_str}: {e_save_p}")
            # This individual file save error contributes to overall dataset processing status
            dataset_succeeded = False
            # Log this specific file error if needed for failed_dataset_list,
            # but main_failed_dataset_list is for dataset-level summary.
            # The original script added the *dataset path* to failed_dataset_list if process_parquet returned False.
            # Here, if even one parquet file fails to save, we might consider the dataset processing "failed".

    # --- After all Parquet files in the dataset are processed ---
    if not dataset_succeeded or num_parquet_files_successfully_processed < len(parquet_files_to_process):
        logging.error(f"Dataset {current_dataset_path_obj.name} processing encountered errors or did not process all files.")
        print(f"Dataset {current_dataset_path_obj.name} processing encountered errors or did not process all files.")
        # Replicating original: add to failed_dataset_list.
        # The 'features' part in original was bboxes at the point of failure. Here, it's more complex.
        # Let's store the path and a reason. The bboxes collected so far are in dataset_level_all_bboxes.
        main_failed_dataset_list.append({
            "path": dataset_path_str, # Original user-provided path
            "effective_path": str(effective_dataset_path),
            "reason": "One or more Parquet files failed to process or save.",
            "collected_bboxes_count": len(dataset_level_all_bboxes) # For reference
        })
        # Original returned if not finished. We let it continue to save bboxes if any.
    
    # Save aggregated bboxes for this dataset (even if some files failed, save what we have)
    if dataset_level_all_bboxes:
        bboxes_json_out_path = effective_dataset_path / "meta" / "bboxes.jsonl"
        try:
            df_bboxes_dataset = pandas.DataFrame(dataset_level_all_bboxes)
            df_bboxes_dataset.to_json(bboxes_json_out_path, orient="records", lines=True, force_ascii=False)
            logging.info(f"Bboxes for dataset {current_dataset_path_obj.name} saved to {bboxes_json_out_path}")
        except Exception as e_save_bbox_json:
            logging.error(f"Failed to save bboxes.jsonl for {current_dataset_path_obj.name}: {e_save_bbox_json}")
            # Add to failed_dataset_list if this is considered a dataset failure criteria
            if dataset_succeeded: # If it was fine until now
                 main_failed_dataset_list.append({
                    "path": dataset_path_str, "reason": f"Failed to save bboxes.jsonl: {e_save_bbox_json}"
                 })

    if dataset_succeeded and num_parquet_files_successfully_processed == len(parquet_files_to_process):
        logging.info(f"Successfully processed dataset {current_dataset_path_obj.name}.")
    else:
        logging.warning(f"Dataset {current_dataset_path_obj.name} processed with issues.")


def main_entry_point_final():
    global main_failed_dataset_list, OVERWRITE_ORIGINAL_DATASET_BEHAVIOR

    parser = argparse.ArgumentParser(description="Parallel Grounding with Persistent Pool & Full Feature Parity.")
    parser.add_argument("--dataset-path", type=str, help="Path of a single LeRobot dataset.")
    parser.add_argument("--dataset-home", type=str, help="Home directory of multiple LeRobot datasets.")
    parser.add_argument("--output-home", type=str, help="Output directory (required if not --inplace).")
    parser.add_argument("--inplace", type=int, default=0, choices=[0, 1], help="1 to overwrite, 0 to copy.")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of worker processes in the pool.")
    
    cli_args = parser.parse_args()

    # Set overwrite behavior based on args
    if cli_args.inplace == 1:
        OVERWRITE_ORIGINAL_DATASET_BEHAVIOR = True
        logging.warning("IN-PLACE processing enabled. Original datasets will be modified.")
        if cli_args.output_home:
            logging.warning("--output-home is specified but will be IGNORED due to --inplace=1.")
            cli_args.output_home = None # Nullify to prevent misuse
    else:
        OVERWRITE_ORIGINAL_DATASET_BEHAVIOR = False
        if not cli_args.output_home:
            logging.error("Must specify --output-home if not using --inplace=1.")
            sys.exit(1)

    # Adjust num_proc
    if cli_args.num_proc <= 0:
        cli_args.num_proc = mp.cpu_count() or 1 # Default to all CPUs if 0 or less
        logging.info(f"Auto-set num-proc to {cli_args.num_proc} (os.cpu_count).")

    # --- Create the persistent process pool ---
    # 'spawn' context is critical for CUDA safety with multiprocessing.
    # The initializer loads the model in each worker.
    mp_spawn_context = mp.get_context('spawn')
    with mp_spawn_context.Pool(processes=cli_args.num_proc,
                               initializer=init_worker_process_for_pool,
                               maxtasksperchild=None) as process_pool: # `None` means workers live as long as pool
        logging.info(f"Persistent process pool with {cli_args.num_proc} workers (spawn context) created and initialized.")

        if cli_args.dataset_path:
            logging.info(f"Processing single dataset: {cli_args.dataset_path}")
            process_single_dataset_final(cli_args.dataset_path, cli_args, process_pool)
        elif cli_args.dataset_home:
            dataset_home_p_obj = Path(cli_args.dataset_home)
            if not dataset_home_p_obj.is_dir():
                logging.error(f"Dataset home {dataset_home_p_obj} is not a valid directory.")
                sys.exit(1)
            
            all_dataset_paths_in_home = [str(d) for d in dataset_home_p_obj.iterdir() if d.is_dir()]
            logging.info(f"Found {len(all_dataset_paths_in_home)} potential datasets in {dataset_home_p_obj}.")
            
            for i, single_ds_path_str in enumerate(all_dataset_paths_in_home):
                logging.info(f"--- Starting dataset {i+1}/{len(all_dataset_paths_in_home)}: {Path(single_ds_path_str).name} ---")
                process_single_dataset_final(single_ds_path_str, cli_args, process_pool)
                logging.info(f"--- Finished dataset {i+1}/{len(all_dataset_paths_in_home)}: {Path(single_ds_path_str).name} ---")
        else:
            logging.error("Either --dataset-path or --dataset-home must be provided.")
            sys.exit(1)

    # --- Save failed datasets list (if any) ---
    # This replicates the original script's behavior for failed_dataset_list
    if main_failed_dataset_list:
        logging.info(f"There are {len(main_failed_dataset_list)} datasets that encountered issues. Details will be saved to './failed_datasets.jsonl'.")
        failed_datasets_output_path = Path("failed_datasets.jsonl") # Name from original script
        try:
            # The original script used pandas to save this.
            # The structure of items in main_failed_dataset_list should be compatible.
            df_failed_datasets_summary = pandas.DataFrame(main_failed_dataset_list)
            df_failed_datasets_summary.to_json(failed_datasets_output_path, orient="records", lines=True, force_ascii=False)
            logging.info(f"Summary of datasets with issues saved to: {failed_datasets_output_path}")
        except Exception as e_save_failed_log:
            logging.error(f"Could not save the failed datasets summary log: {e_save_failed_log}")
            # Fallback to simple JSON dump if pandas fails
            try:
                with open(failed_datasets_output_path.with_suffix('.fallback.jsonl'), 'w') as f_fallback:
                    for entry in main_failed_dataset_list:
                        json.dump(entry, f_fallback)
                        f_fallback.write('\n')
                logging.info(f"Fallback failure log saved to: {failed_datasets_output_path.with_suffix('.fallback.jsonl')}")
            except: pass # Best effort
    else:
        logging.info("All specified datasets processed (check logs for per-file details if any).")


if __name__ == "__main__":
    # Set TOKENIZERS_PARALLELISM for the main process. Child processes (spawned)
    # will inherit environment variables, but it's good practice.
    # `datasets` library often handles this for its own multiprocessing too.
    if os.environ.get("TOKENIZERS_PARALLELISM") is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logging.info("Set TOKENIZERS_PARALLELISM=false for the environment.")
    
    # No need for mp.set_start_method('spawn') here if using mp.get_context('spawn').Pool
    main_entry_point_final()