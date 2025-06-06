"""
CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=true python batched_grounding.py \
    --dataset-path /pdata/oxe_lerobot/dlr_sara_pour_converted_externally_to_rlds \
    --output-home /pdata/oxe_lerobot_g \
    --batch_size 24

CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=false python batched_grounding.py \
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

SAVE_INSPECTION_IMAGE = True
RANDOM_SAMPLE_RATE = 0.0005 # ratio of images that will be saved for inspection

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

failed_dataset_list = load_list_from_jsonl("failed_datasets.jsonl")
def save_fail_list_to_json():
    global failed_dataset_list
    logging.info(f"There are {len(failed_dataset_list)} datasets failed to process. The paths and features will be saved to './failed_datasets.jsonl'.")
    save_list_to_jsonl(failed_dataset_list, "failed_datasets.jsonl")

def handle_sigint(signum, frame):
    logging.info("Caught Ctrl+C, saving list before exit...")
    save_finish_list_to_json()
    save_fail_list_to_json()
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

def process_parquet(parquet_paths, tasks=None, bboxes_json_path:Path=None, dataset_tag=""):
    global finished_parquet_list
    global failed_dataset_list

    # To store {'bbox_index': X, 'bbox': Y} for the entire dataset
    if bboxes_json_path and bboxes_json_path.exists():
        all_episode_bboxes_data = load_list_from_jsonl(str(bboxes_json_path))
    else:
        all_episode_bboxes_data = []

    tasks_available = True
    if USE_SUBTASK_CONDITIONING:
        if tasks is None:
            logging.warning(f"No tasks provided. Disabling subtask conditioning for all files.")
            tasks_available = False

    for parquet_path_str in tqdm(parquet_paths, desc=f"Processing Parquet Files", unit="files"):
        parquet_path = Path(parquet_path_str) # Ensure it's a Path object
        if str(parquet_path) in finished_parquet_list:
            logging.info(f"Skipping already processed file: {parquet_path_str}")
            continue

        try:
            episode = load_dataset("parquet", data_files=str(parquet_path), split='train')
            features = episode.features
            logging.debug(f"Loaded dataset from {parquet_path} with {len(episode)} records.")
        except Exception as e:
            logging.error(f"Error loading {parquet_path}: {e}")
            failed_dataset_list.append({"path": str(parquet_path), "reason": f"Load error: {e}", "features": None})
            return False
        
        image_columns_info = [] # List of (col_name, col_type_enum_or_class)
        # suspected_image_columns = [col for col in features if col.startswith("observation.images.cam")]
        suspected_image_columns = [col for col in features if col == "observation.images.cam"]
        for col in suspected_image_columns:
            if isinstance(features[col], ImageHF):
                image_columns_info.append((col, ImageHF))
            elif isinstance(features[col], dict) and 'bytes' in features[col]:
                image_columns_info.append((col, dict)) # Using dict as a sentinel for byte-based images
            # else:
            #     logging.warning(f"Column {col} in {parquet_path} looks like an image column but has unrecognized type: {type(features[col])}")

        if not image_columns_info:
            logging.error(f"No recognized image columns found in {parquet_path}. Skipping.")
            failed_dataset_list.append({"path": str(parquet_path), "reason": "No image columns", "features": features})
            return False

        processed_records_for_this_file = []
        
        # --- Batching Logic ---
        current_batch_images = []
        current_batch_task_descs = []
        current_batch_metadata = [] # To store (record_idx_in_episode, image_col_name, image_col_type)

        temp_episode_records = list(episode) # Convert to list for easier indexing if needed later

        for r_idx, record in tqdm(enumerate(temp_episode_records), desc=f" Records", total=len(temp_episode_records), leave=False):
            r_task_index = record.get('task_index', None)
            task_str = None
            
            final_use_task_for_record = USE_SUBTASK_CONDITIONING and tasks_available
            if final_use_task_for_record:
                if 'task_index' not in features:
                    # This warning will print once per file if task_index is missing
                    if r_idx == 0: logging.warning(f"No task_index column in {parquet_path}. Disabling subtask conditioning for this file.")
                    final_use_task_for_record = False
                elif r_task_index is not None:
                    try:
                        task_row = tasks[tasks['task_index'] == r_task_index]
                        if not task_row.empty:
                            task_str = task_row.iloc[0]['task']
                        else:
                            # This warning will print for each record missing a task_index map
                            logging.warning(f"Task_index {r_task_index} not found in tasks.jsonl for {parquet_path}, record {r_idx}. Disabling subtask for this record.")
                            final_use_task_for_record = False # Disable for this specific record's images
                    except Exception as e:
                        logging.warning(f"Error retrieving task for task_index {r_task_index} in {parquet_path}, record {r_idx}: {e}. Disabling subtask for this record.")
                        final_use_task_for_record = False # Disable for this specific record's images
                else: # r_task_index is None
                    if r_idx == 0: logging.info(f"Record {r_idx} in {parquet_path} has None for task_index. Disabling subtask for its images.")
                    final_use_task_for_record = False


            for image_col_name, image_col_type in image_columns_info:
                try:
                    if image_col_type == ImageHF:
                        pil_image = record[image_col_name]
                        if pil_image is None: # Skip if image is None
                            logging.warning(f"Image is None for record {r_idx}, column {image_col_name} in {parquet_path}. Skipping.")
                            continue
                    elif image_col_type == dict: # Bytes image
                        img_bytes_data = record[image_col_name]
                        if img_bytes_data is None or img_bytes_data.get('bytes') is None:
                             logging.warning(f"Image bytes data is None for record {r_idx}, column {image_col_name} in {parquet_path}. Skipping.")
                             continue
                        pil_image = Image.open(io.BytesIO(img_bytes_data['bytes']))
                    else: # Should not happen given earlier checks
                        logging.error(f"Unexpected image type {image_col_type} for column {image_col_name}")
                        continue
                    
                    current_batch_images.append(pil_image)
                    # ###
                    # # debug: save img to inspection_images/debug/
                    # pil_image.save(f"inspection_images/debug/{r_idx}_{image_col_name}_o.jpg")
                    # ###
                    current_batch_task_descs.append(task_str if final_use_task_for_record else None)
                    current_batch_metadata.append({
                        "original_record_idx": r_idx, # Index within the current parquet file's episode
                        "image_col_name": image_col_name,
                        "image_col_type": image_col_type,
                        "task_str_used": task_str if final_use_task_for_record else None # For inspection image saving
                    })
                except Exception as e:
                    logging.error(f"Error processing image for record {r_idx}, col {image_col_name} in {parquet_path}: {e}")
                    # Decide how to handle: skip this image, skip record, or skip file.
                    # For now, let's skip this image and continue batching others.
                    continue


                if len(current_batch_images) >= BATCH_SIZE:
                    # Process the current batch
                    logging.debug(f"Processing batch of size {len(current_batch_images)}")
                    batched_json_responses, _ = grounding_pipeline_batched(
                        images=current_batch_images,
                        model=q_model,
                        processor=q_processor,
                        task_descs=current_batch_task_descs,
                        SHOW_OBJECT_NAME=SHOW_OBJECT_NAME,
                        USE_SUBTASK_CONDITIONING=USE_SUBTASK_CONDITIONING # This is now handled per-image by task_descs
                    )

                    # Distribute results back to records
                    for i in range(len(batched_json_responses)):
                        meta = current_batch_metadata[i]
                        record_to_update = temp_episode_records[meta["original_record_idx"]]
                        # output_image = batched_output_images[i]
                        json_response = batched_json_responses[i]

                        if SAVE_INSPECTION_IMAGE and (random.random() < RANDOM_SAMPLE_RATE):
                            global inspection_image_id
                            inspection_image_id += 1
                            parquet_info_str = (f"{dataset_tag}_{meta['image_col_name'][-1]}").replace('/', '-').replace('\\', '-').replace('.', '-')
                            # output_image.save(f"inspection_images/{inspection_image_id}_{parquet_info_str}.jpg")
                            save_json_infos = {
                                'parquet_path': str(parquet_path),
                                'record_idx': meta["original_record_idx"],
                                'image_col_name': meta['image_col_name'],
                                'task_desc': meta['task_str_used'], # Use the actual task string passed
                                'json_response': json_response
                            }
                            with open(f"inspection_images/{inspection_image_id}_{parquet_info_str}.json", 'w') as json_file:
                                json.dump(save_json_infos, json_file, ensure_ascii=False, indent=4)

                        # if meta["image_col_type"] == ImageHF:
                        #     record_to_update[f"{meta['image_col_name']}_g"] = output_image
                        # elif meta["image_col_type"] == dict:
                        #     buffer = io.BytesIO()
                        #     img_format = output_image.format or "PNG"
                        #     output_image.save(buffer, format=img_format)
                        #     record_to_update[f"{meta['image_col_name']}_g"] = record_to_update[meta["image_col_name"]].copy()
                        #     record_to_update[f"{meta['image_col_name']}_g"]['bytes'] = buffer.getvalue()

                        if 'bbox_index' not in record_to_update: # Only add once per record
                            bbox_idx = len(all_episode_bboxes_data)
                            all_episode_bboxes_data.append({
                                'bbox_index': bbox_idx,
                                'bbox': json_response # This is the json_response for *this specific image*
                            })
                            # record_to_update['bbox_index'] = bbox_idx
                            record_to_update['bbox'] = json.dumps(json_response, ensure_ascii=False)
                    
                    # Clear batches
                    current_batch_images = []
                    current_batch_task_descs = []
                    current_batch_metadata = []
        
        # Process any remaining images in the last batch
        if current_batch_images:
            logging.debug(f"Processing final batch of size {len(current_batch_images)}")
            batched_json_responses, _ = grounding_pipeline_batched(
                images=current_batch_images, model=q_model, processor=q_processor,
                task_descs=current_batch_task_descs, SHOW_OBJECT_NAME=SHOW_OBJECT_NAME,
                USE_SUBTASK_CONDITIONING=USE_SUBTASK_CONDITIONING # Handled by task_descs
            )
            for i in range(len(batched_json_responses)): # Same logic as above
                meta = current_batch_metadata[i]
                record_to_update = temp_episode_records[meta["original_record_idx"]]
                # output_image = batched_output_images[i]
                json_response = batched_json_responses[i]

                if SAVE_INSPECTION_IMAGE and (random.random() < RANDOM_SAMPLE_RATE):
                    inspection_image_id += 1
                    parquet_info_str = (f"{dataset_tag}_{meta['image_col_name'][-1]}").replace('/', '-').replace('\\', '-').replace('.', '-')
                    # output_image.save(f"inspection_images/{inspection_image_id}_{parquet_info_str}.jpg")
                    save_json_infos = {
                        'parquet_path': str(parquet_path), 'record_idx': meta["original_record_idx"],
                        'image_col_name': meta['image_col_name'], 'task_desc': meta['task_str_used'],
                        'json_response': json_response
                    }
                    with open(f"inspection_images/{inspection_image_id}_{parquet_info_str}.json", 'w') as json_file:
                        json.dump(save_json_infos, json_file, ensure_ascii=False, indent=4)

                # if meta["image_col_type"] == ImageHF:
                #     record_to_update[f"{meta['image_col_name']}_g"] = output_image
                # elif meta["image_col_type"] == dict:
                #     buffer = io.BytesIO()
                #     img_format = output_image.format or "PNG"
                #     output_image.save(buffer, format=img_format)
                #     record_to_update[f"{meta['image_col_name']}_g"] = record_to_update[meta["image_col_name"]].copy() # Copy original dict structure
                #     record_to_update[f"{meta['image_col_name']}_g"]['bytes'] = buffer.getvalue()
                
                if 'bbox_index' not in record_to_update:
                    bbox_idx = len(all_episode_bboxes_data)
                    all_episode_bboxes_data.append({'bbox_index': bbox_idx, 'bbox': json_response})
                    # record_to_update['bbox_index'] = bbox_idx
                    record_to_update['bbox'] = json.dumps(json_response, ensure_ascii=False)
        
        # All records for this file have been (potentially) updated in temp_episode_records
        processed_records_for_this_file = temp_episode_records

        # Save updated parquet
        new_features = features.copy()
        # if 'bbox_index' not in new_features: # Add if not already there from a previous run
        #     new_features['bbox_index'] = Value('int64')
        if 'bbox' not in new_features:
            new_features['bbox'] = Value("string")
        # for image_col_name, image_col_type in image_columns_info:
        #     if f"{image_col_name}_g" not in new_features:
        #         new_features[f"{image_col_name}_g"] = features[image_col_name] # Keep the original type
        
        if processed_records_for_this_file:
            updated_dataset = Dataset.from_list(processed_records_for_this_file, features=new_features)
        else:
            logging.error(f"No records processed for {parquet_path}, creating empty dataset with new features.")
            failed_dataset_list.append({"path": str(parquet_path), "reason": f"Process error: 'processed_records_for_this_file' is empty", "features": new_features})
            return False
        
        try:
            updated_dataset.to_parquet(str(parquet_path))
            save_list_to_jsonl(all_episode_bboxes_data, str(bboxes_json_path))
            save_finish_list_to_json()
            # logging.info(f"Successfully processed and saved {parquet_path}")
            # print(f"Successfully processed and saved {parquet_path}.\nUpdated {bboxes_json_path} with {len(all_episode_bboxes_data)} items.")
            finished_parquet_list.append(str(parquet_path))
        except Exception as e:
            logging.error(f"Error saving updated dataset to {parquet_path}: {e}")
            failed_dataset_list.append({"path": str(parquet_path), "reason": f"Save error: {e}", "features": new_features})
            return False


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
        tasks = pandas.read_json(task_json_path, lines=True)
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
    atexit.register(save_fail_list_to_json)
    signal.signal(signal.SIGINT, handle_sigint)
    
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    logging.info(f"Batch size set to {BATCH_SIZE}.")

    q_model, q_processor = setup_model(MODEL_NAME)
    
    main(args)