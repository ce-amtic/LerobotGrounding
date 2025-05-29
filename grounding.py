"""
CUDA_VISIBLE_DEVICES=6,7 TOKENIZERS_PARALLELISM=true python grounding.py \
    --dataset-path /pdata/oxe_lerobot/dlr_sara_pour_converted_externally_to_rlds \
    --output-home /pdata/oxe_lerobot_g > o.txt

CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=true python grounding.py \
    --dataset-home /pdata/oxe_lerobot \
    --finished-list-path finished_parquets.json \
    --inplace 1 > o.txt
"""

import os, sys, io, signal
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


### --- qwen pipeline --- ###
from qwen_grounding_utils import setup_model, grounding_pipeline
q_model, q_processor = setup_model(MODEL_NAME)

inspection_image_id = 0
os.makedirs("inspection_images", exist_ok=True)
failed_dataset_list = []
finished_parquet_list = ['first', 'second']


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

def process_parquet(parquet_paths, tasks=None):
    bboxes = [] ### output ###

    tasks_available = True
    if USE_SUBTASK_CONDITIONING:
        if tasks is None: 
            logging.warning(f"No tasks provided for {parquet_path}. Disabling subtask conditioning.")
            tasks_available = False

    for parquet_path in tqdm(parquet_paths, desc="Processing Parquet Files", unit="files"):
        if str(parquet_path) in finished_parquet_list:
            continue

        try:
            episode = load_dataset("parquet", data_files=str(parquet_path, ), split='train')
            features = episode.features
            logging.debug(f"Loaded dataset from {parquet_path} with {len(episode)} records.")
            logging.debug(f"Columns: {features}")
        except Exception as e:
            logging.error(f"Error loading {parquet_path}: {e}")
        
        # image_columns = [col for col in features if isinstance(features[col], ImageHF)]
        # improve auto-detection of image columns
        image_columns = []
        suspected_image_columns = [col for col in features if col.startswith("observation.images.cam")]
        for col in suspected_image_columns:
            if isinstance(features[col], ImageHF):
                image_columns.append((col, ImageHF))
            elif isinstance(features[col], dict) and 'bytes' in features[col]:
                image_columns.append((col, dict))
            else:
                logging.error(f"unrecognized image type: {type(features[col])}")
                # try:
                #     episode = episode.cast_column(col, ImageHF())
                #     image_columns.append(col)
                # except Exception as e:
                #     logging.error(f"unrecognized image type: {type(features[col])}")

        if not image_columns:
            logging.error("Failed to auto-detect image columns. Please check the dataset.")
            return False, features

        processed_records_for_this_file = [] ### output ###

        task_index_available = True
        if USE_SUBTASK_CONDITIONING and tasks_available:
            if 'task_index' not in features:
                logging.warning(f"No task_index found in {parquet_path}. Disabling subtask conditioning.")
                task_index_available = False

        for r_idx, record in tqdm(enumerate(episode), desc="Processing Record", total=len(episode)): # record is a dict
            r_task_index = record.get('task_index', None)
            task_index_check = True
            if USE_SUBTASK_CONDITIONING and tasks_available and task_index_available:
                try:
                    task_row = tasks[tasks['task_index'] == r_task_index]
                    task_str = task_row.iloc[0]['task']
                except Exception as e:
                    logging.warning(f"Error retrieving task for task_index {r_task_index}: {e}. Disabling subtask conditioning.")
                    task_index_check = False
            
            final_use_task = USE_SUBTASK_CONDITIONING and tasks_available and task_index_available and task_index_check
            r_bbox = None
            for image_col, image_type in image_columns:
                logging.debug(f"Processing record[{r_idx}] in {parquet_path}, image column: {image_col}")

                if image_type == ImageHF:
                    image = record[image_col]
                elif image_type == dict:
                    image = Image.open(io.BytesIO(record[image_col]['bytes']))

                json_response, output_image = grounding_pipeline(
                    image=image,
                    model=q_model,
                    processor=q_processor,
                    task_desc=task_str if final_use_task else None,
                    SHOW_OBJECT_NAME=SHOW_OBJECT_NAME,
                    USE_SUBTASK_CONDITIONING=final_use_task
                )

                if SAVE_INSPECTION_IMAGE and (random.random() < RANDOM_SAMPLE_RATE):
                    global inspection_image_id
                    inspection_image_id += 1
                    parquet_info_str = str(parquet_path).replace('/', '-').replace('\\', '-').replace('.', '-')
                    output_image.save(f"inspection_images/{inspection_image_id}_{parquet_info_str}.jpg")
                    save_json_infos = {
                        'parquet_path': str(parquet_path),
                        'task_desc': task_str if final_use_task else '',
                        'json_response': json_response
                    }
                    with open(f"inspection_images/{inspection_image_id}_{parquet_info_str}.json", 'w') as json_file:
                        json.dump(save_json_infos, json_file, ensure_ascii=False, indent=4)

                if image_type == ImageHF:
                    record[image_col] = output_image
                elif image_type == dict:
                    buffer = io.BytesIO()
                    output_image.save(buffer, format=output_image.format or "PNG")
                    record[image_col]['bytes'] = buffer.getvalue()
                if r_bbox == None: r_bbox = json_response
            
            r_bbox_index = len(bboxes)
            bboxes.append({
                'bbox_index': r_bbox_index,
                'bbox': r_bbox
            })
            record['bbox_index'] = r_bbox_index

            processed_records_for_this_file.append(record)

        ### output: save new parquet ###
        new_features = features.copy()
        new_features['bbox_index'] = Value('int64')
        if processed_records_for_this_file:
            updated_dataset = Dataset.from_list(processed_records_for_this_file, features=new_features)
        else:
            updated_dataset = Dataset.from_list([], features=new_features)
        try:
            updated_dataset.to_parquet(str(parquet_path))
        except Exception as e:
            logging.error(f"Error saving updated dataset to {parquet_path}: {e}")
            print(f"Error saving updated dataset to {parquet_path}: {e}")
            return False, new_features
        ######
        print(f"Processed parquet {str(parquet_path)} successfully")
        finished_parquet_list.append(str(parquet_path))

    return True, bboxes


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
    finished, bboxes = process_parquet(load_parquet(output_dataset_path), tasks=tasks)
    if not finished:
        logging.error(f"Failed to process dataset {dataset_path}. Please check the logs for details.")
        failed_dataset_list.append({
            "path": dataset_path,
            "features": bboxes
        })
        return

    # save bboxes to bboxes.jsonl
    bboxes_json_path = output_dataset_path / "meta" / "bboxes.jsonl"
    df = pandas.DataFrame(bboxes)
    df.to_json(bboxes_json_path, orient="records", lines=True, force_ascii=False)

    print(f"Processed dataset {dataset_path} successfully. Bboxes saved to {bboxes_json_path}.")


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

    # save failed datasets if any
    if failed_dataset_list:
        logging.info(f"There are {len(failed_dataset_list)} datasets failed to process. The paths and features will be saved to './failed_datasets.jsonl'.")
        failed_datasets_path = Path("failed_datasets.jsonl")
        df_failed = pandas.DataFrame(failed_dataset_list)
        df_failed.to_json(failed_datasets_path, orient="records", lines=True, force_ascii=False) 

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
    args = parser.parse_args()
    
    if args.inplace == 1:
        if not args.finished_list_path:
            logging.error("Provide finish_parquets.json through --finished-list-path to enable inplace modification.")
            exit(0)
        OVERWRITE_ORIGINAL_DATASET = True
        logging.warning("Output home directory is not specified. Will overwrite the original dataset.")

        with open(args.finished_list_path, "r", encoding="utf-8") as f:
            finished_parquet_list = json.load(f)

        def save_list_to_json():
            with open(args.finished_list_path, "w", encoding="utf-8") as f:
                json.dump(finished_parquet_list, f, ensure_ascii=False, indent=2)
        atexit.register(save_list_to_json)

        def handle_sigint(signum, frame):
            logging.info("Caught Ctrl+C, saving list before exit...")
            save_list_to_json()
            sys.exit(0)
        signal.signal(signal.SIGINT, handle_sigint)
    
    main(args)