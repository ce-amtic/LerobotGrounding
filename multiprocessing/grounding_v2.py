"""
CUDA_VISIBLE_DEVICES=6,7 TOKENIZERS_PARALLELISM=false python grounding_v2.py \
    --dataset-path /pdata/oxe_lerobot/dlr_sara_pour_converted_externally_to_rlds \
    --output-home /pdata/oxe_lerobot_g \
    --num-proc 8 > o.txt

CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=false python grounding_v2.py \
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
from datasets import load_dataset, Dataset, Value, Features
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

# --- Global State (managed carefully for parallelism) ---
# q_model_global and q_processor_global will NOT be used by spawned child processes directly.
# Each child process will load its own instance.
# q_model_global = None # No longer needed as a globally shared instance for workers
# q_processor_global = None

inspection_image_id_counter = 0
os.makedirs("inspection_images", exist_ok=True)
failed_dataset_processing_log = []

IMAGE_TYPE_HF_ENUM = 1
IMAGE_TYPE_DICT_BYTES_ENUM = 2

# --- Per-Worker Model Cache (to avoid reloading in the same worker process if map calls fn multiple times) ---
# This is a dictionary unique to each worker process
_worker_model_cache = {}

def get_qwen_pipeline_components_for_worker():
    """
    Loads or retrieves Qwen model and processor for the current worker process.
    Caches the loaded model within the worker process to avoid redundant loading.
    """
    global _worker_model_cache # This global is specific to each worker process
    from qwen_grounding_utils import setup_model # Import here for worker

    pid = os.getpid()
    if pid not in _worker_model_cache:
        logging.info(f"Worker {pid}: Initializing Qwen model ({MODEL_NAME_CONFIG})...")
        try:
            model, processor = setup_model(MODEL_NAME_CONFIG)
            _worker_model_cache[pid] = (model, processor)
            logging.info(f"Worker {pid}: Qwen model initialized.")
        except Exception as e_setup:
            logging.error(f"Worker {pid}: Failed to setup model: {e_setup}")
            raise # Re-raise to stop this worker if model setup fails
    else:
        # logging.debug(f"Worker {pid}: Using cached Qwen model.") # Can be too verbose
        pass
    return _worker_model_cache[pid]


def _get_image_columns_config(features_obj: Features):
    image_columns = []
    suspected_cam_cols = [col for col in features_obj if col.startswith("observation.images.cam")]
    other_cols = [col for col in features_obj if not col.startswith("observation.images.cam")]
    for col_name in suspected_cam_cols + other_cols:
        feature_type = features_obj[col_name]
        if isinstance(feature_type, ImageHF):
            image_columns.append((col_name, IMAGE_TYPE_HF_ENUM))
        elif isinstance(feature_type, dict) and 'bytes' in feature_type:
            if col_name.startswith("observation.images.cam"):
                image_columns.append((col_name, IMAGE_TYPE_DICT_BYTES_ENUM))
    return image_columns

def _map_process_single_record(record_dict, # Single record
                               # Kwargs from fn_kwargs
                               tasks_df_for_map=None,
                               image_columns_config_for_map=None,
                               task_index_col_exists_for_map=False,
                               current_parquet_path_for_map="",
                               show_object_name_map=False,
                               use_subtask_cond_map=False,
                               save_inspection_img_map=False,
                               random_sample_rate_map=0.0
                               ):
    from qwen_grounding_utils import grounding_pipeline # Import here for worker
    
    # Get model components for this worker (loads/caches them per worker process)
    model_to_use, processor_to_use = get_qwen_pipeline_components_for_worker()

    processed_record = dict(record_dict)
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
            except Exception:
                pass
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
        except Exception:
            pil_image_input = None

        if pil_image_input is None:
            if record_level_bbox_json_response is None:
                record_level_bbox_json_response = {"error": f"Image not available for {img_col_name}"}
            continue

        json_resp_pipeline, output_img_pipeline = grounding_pipeline(
            image=pil_image_input, model=model_to_use, processor=processor_to_use,
            task_desc=task_description_str if final_use_task_for_pipeline else None,
            SHOW_OBJECT_NAME=show_object_name_map, USE_SUBTASK_CONDITIONING=final_use_task_for_pipeline
        )

        if save_inspection_img_map and output_img_pipeline and (random.random() < random_sample_rate_map):
            if inspection_info_to_return is None:
                 # Send PIL image bytes to avoid pickling issues with complex PIL objects across processes
                buffer = io.BytesIO()
                fmt_insp = output_img_pipeline.format or "JPEG" # JPEG is smaller for inspection
                output_img_pipeline.save(buffer, format=fmt_insp)
                inspection_info_to_return = {
                    'image_bytes_to_save': buffer.getvalue(),
                    'image_format': fmt_insp,
                    'parquet_path': current_parquet_path_for_map,
                    'task_desc': task_description_str if final_use_task_for_pipeline else '',
                    'json_response': json_resp_pipeline,
                }

        if output_img_pipeline is not None:
            if img_type_enum == IMAGE_TYPE_HF_ENUM:
                # To ensure picklability when returning from map, convert PIL to bytes
                # The main process will reconstruct it or save bytes if ImageHF feature is bytes-based.
                # Or, if ImageHF feature can handle PIL directly after map, this is fine.
                # Let's assume ImageHF can handle PIL object for now. If pickling fails, convert to bytes.
                processed_record[img_col_name] = output_img_pipeline
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

    processed_record['_temp_bbox_json_response_'] = record_level_bbox_json_response
    if inspection_info_to_return:
        processed_record['_temp_inspection_info_'] = inspection_info_to_return
    return processed_record


def _process_single_parquet_file_parallel(parquet_file_path_str: str,
                                          tasks_data_frame: pandas.DataFrame = None,
                                          num_map_workers: int = 1):
    global failed_dataset_processing_log
    try:
        episode_ds = load_dataset("parquet", data_files=str(parquet_file_path_str), split='train', keep_in_memory=True)
        original_file_features = episode_ds.features
        logging.debug(f"Loaded {parquet_file_path_str} with {len(episode_ds)} records. Features: {original_file_features}")
    except Exception as e_load:
        logging.error(f"Error loading Parquet file {parquet_file_path_str}: {e_load}")
        failed_dataset_processing_log.append({"path": parquet_file_path_str, "error": str(e_load), "stage": "load_parquet"})
        return None, None, None, None

    img_cols_config = _get_image_columns_config(original_file_features)
    if not img_cols_config:
        logging.warning(f"No image columns auto-detected for {parquet_file_path_str}.")
        failed_dataset_processing_log.append({"path": parquet_file_path_str, "error": "No image columns detected", "stage": "detect_img_cols"})
        return None, None, None, None

    task_idx_col_exists = 'task_index' in original_file_features
    map_fn_kwargs = {
        "tasks_df_for_map": tasks_data_frame, "image_columns_config_for_map": img_cols_config,
        "task_index_col_exists_for_map": task_idx_col_exists, "current_parquet_path_for_map": parquet_file_path_str,
        "show_object_name_map": SHOW_OBJECT_NAME_CONFIG, "use_subtask_cond_map": USE_SUBTASK_CONDITIONING_CONFIG,
        "save_inspection_img_map": SAVE_INSPECTION_IMAGE_CONFIG, "random_sample_rate_map": RANDOM_SAMPLE_RATE_CONFIG
    }

    logging.info(f"Starting .map() for {parquet_file_path_str} with {num_map_workers} worker(s).")
    try:
        # Important: if num_map_workers > 0, datasets will use multiprocessing.
        # The start_method (e.g. 'spawn') should be set in main.
        updated_episode_ds = episode_ds.map(
            _map_process_single_record, batched=False, num_proc=num_map_workers,
            fn_kwargs=map_fn_kwargs, load_from_cache_file=False, # Disable cache for dev
            desc=f"Grounding {Path(parquet_file_path_str).name}"
        )
        logging.info(f"Finished .map() for {parquet_file_path_str}.")
    except Exception as e_map:
        logging.error(f"Error during .map() for {parquet_file_path_str}: {e_map}")
        import traceback
        logging.error(traceback.format_exc())
        failed_dataset_processing_log.append({"path": parquet_file_path_str, "error": str(e_map), "stage": "dataset_map"})
        return None, None, None, None

    final_records_for_this_file = []
    bboxes_for_this_file = []
    inspection_images_to_save_for_this_file = []
    for record_from_map in updated_episode_ds:
        final_record_item = dict(record_from_map)
        temp_bbox_data = final_record_item.pop('_temp_bbox_json_response_', None)
        bboxes_for_this_file.append(temp_bbox_data)
        temp_inspection_info = final_record_item.pop('_temp_inspection_info_', None)
        if temp_inspection_info:
            inspection_images_to_save_for_this_file.append(temp_inspection_info)
        final_records_for_this_file.append(final_record_item)
    return final_records_for_this_file, bboxes_for_this_file, inspection_images_to_save_for_this_file, original_file_features


def process_dataset_parallel(dataset_root_path_str: str, cli_args_main):
    global inspection_image_id_counter, failed_dataset_processing_log
    dataset_path_obj = Path(dataset_root_path_str)
    is_inplace_processing = cli_args_main.inplace == 1

    if is_inplace_processing:
        path_to_process = dataset_path_obj
        logging.info(f"Processing dataset IN-PLACE: {path_to_process}")
    else:
        if not cli_args_main.output_home:
            logging.error("Output home directory is required when not processing in-place.")
            return
        path_to_process = Path(cli_args_main.output_home) / dataset_path_obj.name
        logging.info(f"Output will be in: {path_to_process}")
        try:
            if path_to_process.exists():
                shutil.rmtree(path_to_process)
            shutil.copytree(dataset_path_obj, path_to_process, dirs_exist_ok=False)
            logging.info(f"Copied dataset to {path_to_process}")
        except Exception as e_copy:
            logging.error(f"Failed to copy dataset {dataset_path_obj} to {path_to_process}: {e_copy}")
            failed_dataset_processing_log.append({"path": dataset_root_path_str, "error": str(e_copy), "stage": "copy_dataset"})
            return

    tasks_df = None
    task_json_path = path_to_process / "meta" / "tasks.jsonl"
    if task_json_path.exists():
        try:
            tasks_df = pandas.read_json(task_json_path, lines=True)
        except Exception as e_tasks_load:
            logging.error(f"Failed to load tasks.jsonl from {task_json_path}: {e_tasks_load}")
    elif USE_SUBTASK_CONDITIONING_CONFIG:
        logging.warning(f"tasks.jsonl not found at {task_json_path}. Subtask conditioning will be affected.")

    parquet_files_in_dataset = load_parquet_from_dataset_root(path_to_process)
    if not parquet_files_in_dataset:
        logging.error(f"No Parquet files found in {path_to_process}.")
        failed_dataset_processing_log.append({"path": str(path_to_process), "error": "No parquet files found", "stage": "find_parquets"})
        return

    all_bboxes_for_dataset = []
    num_successful_files = 0
    for p_file_path in tqdm(parquet_files_in_dataset, desc=f"Files in {dataset_path_obj.name}", unit="file"):
        processed_records_list, bboxes_list_from_file, inspection_list_from_file, original_feats = \
            _process_single_parquet_file_parallel(
                p_file_path, tasks_data_frame=tasks_df, num_map_workers=cli_args_main.num_proc
            )
        if processed_records_list is None:
            continue

        current_file_records_with_bbox_index = []
        for i, record_item in enumerate(processed_records_list):
            bbox_data_for_record = bboxes_list_from_file[i] if i < len(bboxes_list_from_file) else None
            current_global_bbox_idx = len(all_bboxes_for_dataset)
            all_bboxes_for_dataset.append({'bbox_index': current_global_bbox_idx, 'bbox': bbox_data_for_record})
            record_item_with_idx = dict(record_item)
            record_item_with_idx['bbox_index'] = current_global_bbox_idx
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
                    except Exception as e_insp_save:
                        logging.error(f"Failed to save inspection image/json: {e_insp_save}")


        final_file_features_dict = {name: f_type for name, f_type in original_feats.items()}
        final_file_features_dict['bbox_index'] = Value('int64')
        img_cols_cfg = _get_image_columns_config(original_feats)
        for img_col_name, img_type_enum in img_cols_cfg:
            if img_type_enum == IMAGE_TYPE_DICT_BYTES_ENUM:
                final_file_features_dict[img_col_name] = original_feats[img_col_name]
            elif img_type_enum == IMAGE_TYPE_HF_ENUM:
                # If record[img_col_name] after map is PIL, ImageHF() is correct feature.
                # If it was converted to bytes for pickling, this needs adjustment or reconversion before from_list.
                # For now, assuming it's PIL or ImageHF can handle what map returns.
                final_file_features_dict[img_col_name] = ImageHF()
        final_file_features_obj = Features(final_file_features_dict)

        dataset_to_save_to_parquet = Dataset.from_list(current_file_records_with_bbox_index or [], features=final_file_features_obj)
        try:
            dataset_to_save_to_parquet.to_parquet(p_file_path)
            num_successful_files += 1
            print(f"Processed parquet {str(p_file_path)} successfully")
        except Exception as e_save_parquet:
            logging.error(f"Error saving processed data to Parquet file {p_file_path}: {e_save_parquet}")
            failed_dataset_processing_log.append({"path": p_file_path, "error": str(e_save_parquet), "stage": "save_parquet"})

    if num_successful_files > 0 and all_bboxes_for_dataset:
        bboxes_json_path = path_to_process / "meta" / "bboxes.jsonl"
        try:
            df_all_bboxes = pandas.DataFrame(all_bboxes_for_dataset)
            df_all_bboxes.to_json(bboxes_json_path, orient="records", lines=True, force_ascii=False)
            logging.info(f"All bboxes for dataset {dataset_path_obj.name} saved to {bboxes_json_path}")
        except Exception as e_save_bboxes_json:
            logging.error(f"Error saving bboxes.jsonl for {dataset_path_obj.name}: {e_save_bboxes_json}")
    elif num_successful_files > 0:
        logging.info(f"Dataset {dataset_path_obj.name} processed, but no bboxes were generated/collected.")


def load_parquet_from_dataset_root(dataset_root_path: Path):
    parquet_paths_found = []
    if not dataset_root_path.is_dir():
        return parquet_paths_found
    data_dir = dataset_root_path / "data"
    if not data_dir.is_dir():
        direct_files = [str(f) for f in dataset_root_path.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        return direct_files
    chunk_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name.startswith("chunk-")]
    if not chunk_folders:
        files_in_data_dir = [str(f) for f in data_dir.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        return files_in_data_dir
    for chunk_dir in chunk_folders:
        files_in_chunk = [str(f) for f in chunk_dir.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        parquet_paths_found.extend(files_in_chunk)
    return parquet_paths_found


def main_entry_point():
    global failed_dataset_processing_log
    parser = argparse.ArgumentParser(description="Parallel Grounding for LeRobot datasets (spawn method).")
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--dataset-home", type=str)
    parser.add_argument("--output-home", type=str)
    parser.add_argument("--inplace", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num-proc", type=int, default=1)
    cli_args = parser.parse_args()

    if cli_args.num_proc == 0:
        cli_args.num_proc = os.cpu_count() or 1
    elif cli_args.num_proc < 0:
        cli_args.num_proc = os.cpu_count() or 1
    logging.info(f"Using num_proc = {cli_args.num_proc} for dataset.map()")

    if cli_args.dataset_path:
        process_dataset_parallel(cli_args.dataset_path, cli_args)
    elif cli_args.dataset_home:
        home_path = Path(cli_args.dataset_home)
        datasets_to_process_paths = [str(d_path) for d_path in home_path.iterdir() if d_path.is_dir()]
        for i, ds_path_str in enumerate(datasets_to_process_paths):
            logging.info(f"--- Starting dataset {i+1}/{len(datasets_to_process_paths)}: {Path(ds_path_str).name} ---")
            process_dataset_parallel(ds_path_str, cli_args)
    else:
        parser.print_help()
        sys.exit(1)

    if failed_dataset_processing_log:
        failed_log_file_path = Path("grounding_parallel_spawn_failures.jsonl")
        with open(failed_log_file_path, 'w') as f_log:
            for entry in failed_dataset_processing_log: json.dump(entry, f_log); f_log.write('\n')
        logging.info(f"Failure log: {failed_log_file_path}")

if __name__ == "__main__":
    # IMPORTANT: Set the start method to 'spawn' for CUDA safety with multiprocessing
    # This must be done in the `if __name__ == "__main__":` block,
    # and before any other multiprocessing or CUDA related calls if possible.
    try:
        mp.set_start_method('spawn', force=True)
        logging.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e_spawn:
        logging.warning(f"Could not set start method to 'spawn' (อาจจะถูกตั้งค่าไปแล้ว หรืออยู่ในสภาพแวดล้อมที่ไม่รองรับ): {e_spawn}")

    if os.environ.get("TOKENIZERS_PARALLELISM") is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main_entry_point()