"""
CUDA_VISIBLE_DEVICES=6,7 TOKENIZERS_PARALLELISM=false python grounding_v3.py \
    --dataset-path /pdata/oxe_lerobot/dlr_sara_pour_converted_externally_to_rlds \
    --output-home /pdata/oxe_lerobot_g \
    --num-proc 8 > o.txt

CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=false python grounding_v3.py \
    --dataset-home /pdata/oxe_lerobot \
    --inplace 1 \
    --num-proc 8 > o.txt
"""

import os
import sys
import io
import argparse
import logging
import pandas
import shutil
import random
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Dataset, Value, Features # Added Features
from datasets.features.image import Image as ImageHF
import multiprocessing as mp # Import multiprocessing

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if os.getenv("DEBUG"):
    logging.getLogger().setLevel(logging.DEBUG)

# --- Global Configuration ---
SHOW_OBJECT_NAME_CONFIG = False
USE_SUBTASK_CONDITIONING_CONFIG = True
SAVE_INSPECTION_IMAGE_CONFIG = True
RANDOM_SAMPLE_RATE_CONFIG = 0.0005
MODEL_NAME_CONFIG = "/fyh/.env/models/Qwen2.5-VL-7B-Instruct"

# --- Global State for Worker Processes (will be set by initializer) ---
# These will be specific to each worker process in the pool
worker_q_model = None
worker_q_processor = None

# --- Other Global State (managed by main process) ---
inspection_image_id_counter = 0
os.makedirs("inspection_images", exist_ok=True)
failed_dataset_processing_log = []

IMAGE_TYPE_HF_ENUM = 1
IMAGE_TYPE_DICT_BYTES_ENUM = 2

# --- Initializer function for multiprocessing.Pool workers ---
def init_worker_process():
    """
    Initializer for each worker process in the Pool.
    Loads the model and processor once per worker process.
    """
    global worker_q_model, worker_q_processor
    from qwen_grounding_utils import setup_model # Import here

    pid = os.getpid()
    logging.info(f"Worker {pid}: Initializing Qwen model ({MODEL_NAME_CONFIG})...")
    try:
        # Load the model and store it in globals specific to this worker process
        worker_q_model, worker_q_processor = setup_model(MODEL_NAME_CONFIG)
        logging.info(f"Worker {pid}: Qwen model initialized and ready.")
    except Exception as e_setup:
        logging.error(f"Worker {pid}: CRITICAL - Failed to setup model: {e_setup}")
        # If a worker can't initialize, it's problematic.
        # The pool might hang or tasks sent to it might fail.
        # Consider a way to signal this failure back or have the worker exit.
        # For now, re-raise to make it visible.
        raise RuntimeError(f"Worker {pid} failed to initialize model") from e_setup

# --- Worker function to be executed by processes in the Pool ---
def apply_grounding_to_record_task(task_args):
    """
    Processes a single record using the pre-loaded model in the worker.
    task_args is a tuple or dict containing all necessary info for one record.
    """
    global worker_q_model, worker_q_processor # Access model loaded by init_worker_process
    from qwen_grounding_utils import grounding_pipeline # Import here

    if worker_q_model is None or worker_q_processor is None:
        # This should not happen if init_worker_process was successful
        logging.error(f"Worker {os.getpid()}: Model not loaded! Skipping task.")
        # Return a result indicating failure or the original record unmodified
        # For now, let's make it clear something went wrong.
        return {"error": "Model not loaded in worker", "original_record": task_args.get("record_dict")}


    # Unpack arguments passed for this specific task
    record_dict = task_args["record_dict"]
    tasks_df_for_map = task_args["tasks_df_for_map"]
    image_columns_config_for_map = task_args["image_columns_config_for_map"]
    task_index_col_exists_for_map = task_args["task_index_col_exists_for_map"]
    current_parquet_path_for_map = task_args["current_parquet_path_for_map"]
    # Unpack config flags
    show_object_name_map = task_args["show_object_name_map"]
    use_subtask_cond_map = task_args["use_subtask_cond_map"]
    save_inspection_img_map = task_args["save_inspection_img_map"]
    random_sample_rate_map = task_args["random_sample_rate_map"]


    processed_record = dict(record_dict) # Mutable copy
    r_task_index_val = processed_record.get('task_index', None)
    task_description_str = None
    can_use_task_for_this_record = False

    if use_subtask_cond_map and tasks_df_for_map is not None and task_index_col_exists_for_map:
        if r_task_index_val is not None:
            try:
                task_row = tasks_df_for_map[tasks_df_for_map['task_index'] == r_task_index_val]
                if not task_row.empty:
                    task_description_str = task_row.iloc[0]['task']
                    can_use_task_for_this_record = True
            except Exception: pass # Keep can_use_task_for_this_record as False
    final_use_task_for_pipeline = use_subtask_cond_map and can_use_task_for_this_record

    record_level_bbox_json_response = None
    inspection_info_to_return = None

    for img_col_name, img_type_enum in image_columns_config_for_map:
        image_data_from_record = processed_record.get(img_col_name)
        pil_image_input = None
        try:
            if img_type_enum == IMAGE_TYPE_HF_ENUM:
                if isinstance(image_data_from_record, Image.Image):
                    pil_image_input = image_data_from_record
                elif isinstance(image_data_from_record, dict) and 'bytes' in image_data_from_record and \
                     isinstance(image_data_from_record['bytes'], bytes):
                    pil_image_input = Image.open(io.BytesIO(image_data_from_record['bytes']))
            elif img_type_enum == IMAGE_TYPE_DICT_BYTES_ENUM:
                if isinstance(image_data_from_record, dict) and 'bytes' in image_data_from_record and \
                   isinstance(image_data_from_record['bytes'], bytes):
                    pil_image_input = Image.open(io.BytesIO(image_data_from_record['bytes']))
        except Exception: pil_image_input = None

        if pil_image_input is None:
            if record_level_bbox_json_response is None:
                record_level_bbox_json_response = {"error": f"Image not available for {img_col_name}"}
            continue

        json_resp_pipeline, output_img_pipeline = grounding_pipeline(
            image=pil_image_input, model=worker_q_model, processor=worker_q_processor,
            task_desc=task_description_str if final_use_task_for_pipeline else None,
            SHOW_OBJECT_NAME=show_object_name_map, USE_SUBTASK_CONDITIONING=final_use_task_for_pipeline
        )

        if save_inspection_img_map and output_img_pipeline and (random.random() < random_sample_rate_map):
            if inspection_info_to_return is None:
                buffer = io.BytesIO()
                fmt_insp = output_img_pipeline.format or "JPEG"
                output_img_pipeline.save(buffer, format=fmt_insp)
                inspection_info_to_return = {
                    'image_bytes_to_save': buffer.getvalue(), 'image_format': fmt_insp,
                    'parquet_path': current_parquet_path_for_map,
                    'task_desc': task_description_str if final_use_task_for_pipeline else '',
                    'json_response': json_resp_pipeline,
                }

        if output_img_pipeline is not None:
            if img_type_enum == IMAGE_TYPE_HF_ENUM:
                # For returning from worker, convert PIL to bytes to ensure picklability
                # The main process will decide how to handle it (reconstruct PIL or use bytes)
                img_byte_buffer = io.BytesIO()
                output_img_pipeline.save(img_byte_buffer, format=output_img_pipeline.format or "PNG")
                processed_record[img_col_name] = {'_pil_bytes_': img_byte_buffer.getvalue(), '_pil_format_': output_img_pipeline.format or "PNG"}

            elif img_type_enum == IMAGE_TYPE_DICT_BYTES_ENUM:
                buffer = io.BytesIO()
                fmt = output_img_pipeline.format or "PNG"
                output_img_pipeline.save(buffer, format=fmt)
                if isinstance(processed_record.get(img_col_name), dict):
                    processed_record[img_col_name]['bytes'] = buffer.getvalue()
                else:
                    processed_record[img_col_name] = {'bytes': buffer.getvalue()}
        
        if record_level_bbox_json_response is None:
            record_level_bbox_json_response = json_resp_pipeline

    # We don't add _temp_ fields directly. Instead, the worker returns a tuple/dict of results.
    # The main process will re-assemble.
    result_payload = {
        "processed_record_content": processed_record, # The modified record content
        "bbox_json_response": record_level_bbox_json_response,
        "inspection_info": inspection_info_to_return
    }
    return result_payload


def _get_image_columns_config(features_obj: Features): # Same as before
    image_columns = []
    suspected_cam_cols = [col for col in features_obj if col.startswith("observation.images.cam")]
    other_cols = [col for col in features_obj if not col.startswith("observation.images.cam")]
    for col_name in suspected_cam_cols + other_cols:
        feature_type = features_obj[col_name]
        if isinstance(feature_type, ImageHF):
            image_columns.append((col_name, IMAGE_TYPE_HF_ENUM))
        elif isinstance(feature_type, dict) and 'bytes' in features_obj[col_name]: # Check the feature definition
            if col_name.startswith("observation.images.cam"):
                image_columns.append((col_name, IMAGE_TYPE_DICT_BYTES_ENUM))
    return image_columns


def _process_single_parquet_file_with_pool(pool: mp.Pool, # Pass the existing pool
                                           parquet_file_path_str: str,
                                           tasks_data_frame: pandas.DataFrame = None):
    global failed_dataset_processing_log
    try:
        episode_ds = load_dataset("parquet", data_files=str(parquet_file_path_str), split='train', keep_in_memory=True)
        original_file_features = episode_ds.features
    except Exception as e_load:
        logging.error(f"Pool: Error loading {parquet_file_path_str}: {e_load}")
        failed_dataset_processing_log.append({"path": parquet_file_path_str, "error": str(e_load), "stage": "pool_load_parquet"})
        return None, None, None, None

    img_cols_config = _get_image_columns_config(original_file_features)
    if not img_cols_config:
        logging.warning(f"Pool: No image columns for {parquet_file_path_str}.")
        failed_dataset_processing_log.append({"path": parquet_file_path_str, "error": "No image columns", "stage": "pool_detect_img_cols"})
        return None, None, None, None # Or return original data if that's desired

    task_idx_col_exists = 'task_index' in original_file_features

    # --- Prepare tasks for the pool ---
    # Each task is a dictionary of arguments for apply_grounding_to_record_task
    tasks_for_pool = []
    for record in episode_ds: # Iterating dataset in main process, sending records to pool
        task_arg = {
            "record_dict": record, # The raw record dict
            "tasks_df_for_map": tasks_data_frame,
            "image_columns_config_for_map": img_cols_config,
            "task_index_col_exists_for_map": task_idx_col_exists,
            "current_parquet_path_for_map": parquet_file_path_str,
            "show_object_name_map": SHOW_OBJECT_NAME_CONFIG,
            "use_subtask_cond_map": USE_SUBTASK_CONDITIONING_CONFIG,
            "save_inspection_img_map": SAVE_INSPECTION_IMAGE_CONFIG,
            "random_sample_rate_map": RANDOM_SAMPLE_RATE_CONFIG
        }
        tasks_for_pool.append(task_arg)

    if not tasks_for_pool: # Empty dataset
        return [], [], [], original_file_features

    # --- Execute tasks in parallel using the pool ---
    # Using imap_unordered for potentially better responsiveness if tasks vary in duration
    # And to process results as they complete.
    # Chunksize can be tuned for performance (balances overhead vs. work per task).
    # A good starting point for chunksize with I/O or mixed workloads can be 1, or a small number.
    # If tasks are very short, larger chunksize. If long (like model inference), smaller.
    # Let's try a sensible chunksize, e.g., number of tasks / (num_workers * 4)
    num_workers_in_pool = pool._processes if hasattr(pool, '_processes') else 1 # Get pool size
    chunk_size = max(1, len(tasks_for_pool) // (num_workers_in_pool * 4)) if num_workers_in_pool > 0 else 1


    logging.info(f"Pool: Submitting {len(tasks_for_pool)} record tasks for {parquet_file_path_str} (chunksize ~{chunk_size}).")
    
    processed_results_from_pool = []
    try:
        # Using tqdm with pool.imap_unordered
        # result_iterator = pool.imap_unordered(apply_grounding_to_record_task, tasks_for_pool, chunksize=chunk_size)
        # for result_payload in tqdm(result_iterator, total=len(tasks_for_pool), desc=f"Pooling {Path(parquet_file_path_str).name}"):
        #     processed_results_from_pool.append(result_payload)
        # Simpler for now, map can also be fine if order doesn't matter for this intermediate step
        processed_results_from_pool = pool.map(apply_grounding_to_record_task, tasks_for_pool, chunksize=chunk_size)

    except Exception as e_pool_map:
        logging.error(f"Pool: Error during task execution for {parquet_file_path_str}: {e_pool_map}")
        import traceback
        logging.error(traceback.format_exc())
        failed_dataset_processing_log.append({"path": parquet_file_path_str, "error": str(e_pool_map), "stage": "pool_task_execution"})
        return None, None, None, None # Hard fail for this file
    
    logging.info(f"Pool: Finished processing {len(processed_results_from_pool)} results for {parquet_file_path_str}.")

    # --- Re-assemble results in the main process ---
    final_records_for_this_file = []
    bboxes_for_this_file = []
    inspection_images_to_save_for_this_file = []

    for result_payload in processed_results_from_pool:
        if isinstance(result_payload, dict) and "error" in result_payload:
            logging.warning(f"Pool: Worker returned error for a record: {result_payload['error']}")
            # Decide how to handle records that failed in worker: skip, use original, etc.
            # For now, we'll just log and it won't contribute to bboxes or inspection.
            # The 'processed_record_content' might be the original if worker failed early.
            # Or it might be partially processed.
            # Let's assume we want to keep the record structure if possible.
            if "processed_record_content" in result_payload :
                 final_records_for_this_file.append(result_payload["processed_record_content"])
            else: # Fallback if structure is entirely lost
                 final_records_for_this_file.append(result_payload.get("original_record", {})) # Or skip
            bboxes_for_this_file.append(None) # Placeholder for bbox
            continue


        record_content = result_payload["processed_record_content"]
        # Handle PIL images returned as bytes
        for img_col_name, img_type_enum in img_cols_config:
            if img_type_enum == IMAGE_TYPE_HF_ENUM and \
               isinstance(record_content.get(img_col_name), dict) and \
               '_pil_bytes_' in record_content[img_col_name]:
                try:
                    pil_bytes_info = record_content[img_col_name]
                    pil_img = Image.open(io.BytesIO(pil_bytes_info['_pil_bytes_']))
                    record_content[img_col_name] = pil_img # Convert back to PIL
                except Exception as e_pil_reconstruct:
                    logging.warning(f"Failed to reconstruct PIL for {img_col_name}: {e_pil_reconstruct}")
                    # Keep as dict or set to None
        
        final_records_for_this_file.append(record_content)
        bboxes_for_this_file.append(result_payload["bbox_json_response"])
        if result_payload["inspection_info"]:
            inspection_images_to_save_for_this_file.append(result_payload["inspection_info"])
            
    return final_records_for_this_file, bboxes_for_this_file, inspection_images_to_save_for_this_file, original_file_features


def process_dataset_with_persistent_pool(dataset_root_path_str: str, cli_args_main, process_pool: mp.Pool):
    global inspection_image_id_counter, failed_dataset_processing_log
    dataset_path_obj = Path(dataset_root_path_str)
    is_inplace_processing = cli_args_main.inplace == 1

    if is_inplace_processing: path_to_process = dataset_path_obj
    else:
        if not cli_args_main.output_home: logging.error("Output home required."); return
        path_to_process = Path(cli_args_main.output_home) / dataset_path_obj.name
        try:
            if path_to_process.exists(): shutil.rmtree(path_to_process)
            shutil.copytree(dataset_path_obj, path_to_process, dirs_exist_ok=False)
        except Exception as e_copy:
            logging.error(f"Failed to copy {dataset_path_obj}: {e_copy}")
            failed_dataset_processing_log.append({"path":dataset_root_path_str, "error":str(e_copy), "stage":"pool_copy"})
            return

    tasks_df = None
    task_json_path = path_to_process / "meta" / "tasks.jsonl"
    if task_json_path.exists():
        try: tasks_df = pandas.read_json(task_json_path, lines=True)
        except Exception as e: logging.error(f"Failed to load {task_json_path}: {e}")
    elif USE_SUBTASK_CONDITIONING_CONFIG: logging.warning(f"{task_json_path} not found.")

    parquet_files_in_dataset = load_parquet_from_dataset_root(path_to_process)
    if not parquet_files_in_dataset:
        logging.error(f"No Parquet files in {path_to_process}.")
        failed_dataset_processing_log.append({"path":str(path_to_process), "error":"No parquet files", "stage":"pool_find_parquets"})
        return

    all_bboxes_for_dataset = []
    num_successful_files = 0
    for p_file_path in tqdm(parquet_files_in_dataset, desc=f"Files in {dataset_path_obj.name}", unit="file"):
        processed_records_list, bboxes_list_from_file, inspection_list_from_file, original_feats = \
            _process_single_parquet_file_with_pool(
                process_pool, p_file_path, tasks_data_frame=tasks_df
            )
        if processed_records_list is None: continue

        current_file_records_with_bbox_index = []
        for i, record_item in enumerate(processed_records_list):
            bbox_data_for_record = bboxes_list_from_file[i] if i < len(bboxes_list_from_file) else None
            current_global_bbox_idx = len(all_bboxes_for_dataset)
            all_bboxes_for_dataset.append({'bbox_index': current_global_bbox_idx, 'bbox': bbox_data_for_record})
            record_item_with_idx = dict(record_item); record_item_with_idx['bbox_index'] = current_global_bbox_idx
            current_file_records_with_bbox_index.append(record_item_with_idx)

        if inspection_list_from_file and SAVE_INSPECTION_IMAGE_CONFIG:
            for insp_info in inspection_list_from_file:
                inspection_image_id_counter += 1
                p_path_str_safe = insp_info['parquet_path'].replace('/', '-').replace('\\', '-').replace('.', '-')
                if insp_info.get('image_bytes_to_save'):
                    try:
                        img_to_save = Image.open(io.BytesIO(insp_info['image_bytes_to_save']))
                        img_to_save.save(f"inspection_images/{inspection_image_id_counter}_{p_path_str_safe}.{insp_info.get('image_format', 'jpg').lower()}")
                        json_to_save = {'original_parquet_path': insp_info['parquet_path'], 'task_desc': insp_info['task_desc'], 'json_response': insp_info['json_response']}
                        with open(f"inspection_images/{inspection_image_id_counter}_{p_path_str_safe}.json", 'w') as json_f:
                            json.dump(json_to_save, json_f, ensure_ascii=False, indent=4)
                    except Exception as e_insp_save: logging.error(f"Failed to save inspection: {e_insp_save}")
        
        final_file_features_dict = {name: f_type for name, f_type in original_feats.items()}
        final_file_features_dict['bbox_index'] = Value('int64')
        img_cols_cfg = _get_image_columns_config(original_feats)
        for img_col_name, img_type_enum in img_cols_cfg:
            if img_type_enum == IMAGE_TYPE_DICT_BYTES_ENUM: final_file_features_dict[img_col_name] = original_feats[img_col_name]
            elif img_type_enum == IMAGE_TYPE_HF_ENUM: final_file_features_dict[img_col_name] = ImageHF()
        final_file_features_obj = Features(final_file_features_dict)

        dataset_to_save = Dataset.from_list(current_file_records_with_bbox_index or [], features=final_file_features_obj)
        try:
            dataset_to_save.to_parquet(p_file_path)
            num_successful_files += 1
            print(f"Processed parquet {str(p_file_path)} successfully")
        except Exception as e: 
            logging.error(f"Error saving {p_file_path}: {e}")
            failed_dataset_processing_log.append({"path":p_file_path, "error":str(e), "stage":"pool_save_parquet"})

    if num_successful_files > 0 and all_bboxes_for_dataset:
        bboxes_json_path = path_to_process / "meta" / "bboxes.jsonl"
        try:
            pandas.DataFrame(all_bboxes_for_dataset).to_json(bboxes_json_path, orient="records", lines=True, force_ascii=False)
        except Exception as e: logging.error(f"Error saving bboxes.jsonl for {dataset_path_obj.name}: {e}")
    elif num_successful_files > 0: logging.info(f"Dataset {dataset_path_obj.name} processed, no bboxes.")


def load_parquet_from_dataset_root(dataset_root_path: Path): # Same as before
    parquet_paths_found = []
    if not dataset_root_path.is_dir(): return parquet_paths_found
    data_dir = dataset_root_path / "data"
    if not data_dir.is_dir():
        return [str(f) for f in dataset_root_path.iterdir() if f.is_file() and f.name.endswith(".parquet")]
    chunk_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name.startswith("chunk-")]
    if not chunk_folders:
        return [str(f) for f in data_dir.iterdir() if f.is_file() and f.name.endswith(".parquet")]
    for chunk_dir in chunk_folders:
        parquet_paths_found.extend([str(f) for f in chunk_dir.iterdir() if f.is_file() and f.name.endswith(".parquet")])
    return parquet_paths_found


def main_entry_point_pool():
    global failed_dataset_processing_log
    parser = argparse.ArgumentParser(description="Persistent Pool Parallel Grounding.")
    parser.add_argument("--dataset-path", type=str); parser.add_argument("--dataset-home", type=str)
    parser.add_argument("--output-home", type=str); parser.add_argument("--inplace", type=int, default=0, choices=[0,1])
    parser.add_argument("--num-proc", type=int, default=1)
    cli_args = parser.parse_args()

    if cli_args.num_proc <= 0: cli_args.num_proc = mp.cpu_count() or 1
    logging.info(f"Using num_proc = {cli_args.num_proc} for persistent ProcessPool.")

    # Create the persistent process pool
    # The initializer will load the model in each worker.
    # `maxtasksperchild` can be useful to recycle workers if they consume too much memory over time.
    # For CUDA, 'spawn' context is crucial.
    ctx = mp.get_context('spawn') # Ensure spawn context
    with ctx.Pool(processes=cli_args.num_proc, initializer=init_worker_process, maxtasksperchild=None) as pool:
        logging.info(f"Persistent process pool with {cli_args.num_proc} workers created and initialized.")
        
        if cli_args.dataset_path:
            process_dataset_with_persistent_pool(cli_args.dataset_path, cli_args, pool)
        elif cli_args.dataset_home:
            home_path = Path(cli_args.dataset_home)
            datasets_paths = [str(d) for d in home_path.iterdir() if d.is_dir()]
            for i, ds_path in enumerate(datasets_paths):
                logging.info(f"--- Pool: Starting dataset {i+1}/{len(datasets_paths)}: {Path(ds_path).name} ---")
                process_dataset_with_persistent_pool(ds_path, cli_args, pool)
        else:
            parser.print_help(); sys.exit(1)

    if failed_dataset_processing_log:
        log_path = Path("grounding_pool_failures.jsonl")
        with open(log_path, 'w') as f:
            for entry in failed_dataset_processing_log: json.dump(entry, f); f.write('\n')
        logging.info(f"Failure log: {log_path}")

if __name__ == "__main__":
    # `set_start_method` should ideally be called only once, and very early.
    # It's already handled by `mp.get_context('spawn')` for the pool.
    # No need to call mp.set_start_method('spawn', force=True) if using get_context.
    if os.environ.get("TOKENIZERS_PARALLELISM") is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main_entry_point_pool()