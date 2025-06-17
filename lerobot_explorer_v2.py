"""
python lerobot_explorer_v2.py \
    --dataset-path /pdata/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds

python lerobot_explorer_v2.py \
    --parquet-file /pdata/oxe_lerobot/bridge_data_v2/data/chunk-000/episode_037744.parquet
"""

import os
import sys
import io
import argparse
import logging
import shutil
import random
import json
from pathlib import Path
from PIL import Image as PILImage # Renamed to avoid conflict with datasets.Image
from tqdm import tqdm
from datasets import load_dataset
from datasets.features.image import Image as HFImage # Hugging Face Image type
from qwen_grounding_utils_v3 import draw_bboxes, json_response_to_bboxes

# --- Configuration ---
LOG_LEVEL = logging.INFO
OUTPUT_IMAGE_DIR = Path("./output_images_explorer")

# --- Setup Logging ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
# set DEBUG=1 to enable debug logs
if os.getenv("DEBUG"):
    logging.getLogger().setLevel(logging.DEBUG)

def user_prompt(message="Continue? (y/n/q to quit all): ", default_yes=False):
    """Prompts the user for input and returns True for 'y', False for 'n', 'q' to quit."""
    while True:
        if default_yes:
            response = input(f"{message} [Y/n/q]: ").strip().lower()
            if response == "" or response == "y":
                return True
            elif response == "n":
                return False
            elif response == "q":
                return "quit"
        else:
            response = input(f"{message} [y/N/q]: ").strip().lower()
            if response == "y":
                return True
            elif response == "" or response == "n":
                return False
            elif response == "q":
                return "quit"
        print("Invalid input. Please enter 'y', 'n', or 'q'.")

def find_parquet_files(dataset_root_path: Path):
    """
    Finds all parquet files within 'chunk-*' subdirectories of the 'data' folder.
    """
    parquet_paths = []
    if not dataset_root_path.exists():
        logging.error(f"Dataset root path {dataset_root_path} does not exist.")
        return parquet_paths

    data_dir = dataset_root_path / "data"
    if not data_dir.exists():
        logging.info(f"'data' directory not found in {dataset_root_path}. Looking for .parquet files directly in root.")
        # Fallback: check for parquet files directly in dataset_root_path
        parquet_files = [str(f) for f in dataset_root_path.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        parquet_paths.extend(parquet_files)
        if not parquet_paths:
             logging.error(f"No .parquet files found directly in {dataset_root_path} either.")
        return parquet_paths


    chunk_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name.startswith("chunk-")]
    if not chunk_folders:
        logging.warning(f"No 'chunk-*' folders found in {data_dir}. Will look for .parquet files directly in {data_dir}.")
        parquet_files_in_data = [str(f) for f in data_dir.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        parquet_paths.extend(parquet_files_in_data)
        if not parquet_files_in_data:
            logging.error(f"No parquet files found in 'chunk-*' folders or directly in {data_dir}.")
        return parquet_paths

    for chunk_folder in chunk_folders:
        parquet_files = [str(f) for f in chunk_folder.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        if not parquet_files:
            logging.warning(f"No parquet files found in {chunk_folder}.")
            continue
        parquet_paths.extend(parquet_files)

    logging.info(f"Found {len(parquet_paths)} parquet files in {dataset_root_path}.")
    return parquet_paths


def explore_parquet_file(parquet_path_str: str, output_base_dir: Path):
    """
    Explores a single parquet file.
    """
    parquet_path = Path(parquet_path_str)
    logging.info(f"\n--- Exploring Parquet File: {parquet_path.name} ---")

    try:
        # Try to load with keep_in_memory, might be faster for smaller files
        dataset = load_dataset("parquet", data_files=parquet_path_str, split="train", keep_in_memory=True)
    except Exception as e_mem:
        logging.warning(f"Failed to load {parquet_path.name} with keep_in_memory=True ({e_mem}). Trying without.")
        try:
            dataset = load_dataset("parquet", data_files=parquet_path_str, split="train")
        except Exception as e:
            logging.error(f"Error loading {parquet_path.name}: {e}")
            return False # Signal to stop or skip

    # 1. Output features
    features = dataset.features
    logging.info(f"Features for {parquet_path.name}:")
    for col_name, col_type in features.items():
        logging.info(f"  - {col_name}: {col_type}")

    # Determine image columns (handles both HFImage and raw bytes in a dict)
    image_columns_info = [] # Stores (column_name, type_enum)
    IMAGE_TYPE_HF = 1
    IMAGE_TYPE_BYTES_DICT = 2

    for col_name, feature_type in features.items():
        if isinstance(feature_type, HFImage):
            image_columns_info.append((col_name, IMAGE_TYPE_HF))
            logging.info(f"Identified HF Image column: {col_name}")
        # Heuristic for Lerobot: observation.images.cam* often stores bytes
        elif col_name.startswith("observation.images.cam") and isinstance(feature_type, dict) and 'bytes' in feature_type:
        # elif col_name == "observation.images.cam" and isinstance(feature_type, dict) and 'bytes' in feature_type:
            # Further check if the 'bytes' field seems like binary data
            if hasattr(feature_type['bytes'], 'pa_type'): # Heuristic: check for pyarrow type info
                 image_columns_info.append((col_name, IMAGE_TYPE_BYTES_DICT))
                 logging.info(f"Identified byte-based Image column (dict): {col_name}")
            else:
                 logging.debug(f"Column {col_name} looks like a dict with 'bytes' but not matching typical binary schema.")
        elif isinstance(feature_type, dict) and 'bytes' in feature_type :
             logging.debug(f"Column {col_name} is a dict with 'bytes', but not named 'observation.images.cam*'. May or may not be an image.")


    if not image_columns_info:
        logging.warning(f"No clearly identifiable image columns found in {parquet_path.name} based on current heuristics.")

    # User prompt to continue to records
    action = user_prompt(f"View records for {parquet_path.name}? (y/n/q): ")
    if action == "quit": return "quit"
    if not action: return True # Continue to next file

    # Sanitize parquet filename for directory creation
    parquet_file_output_dir_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in parquet_path.stem)
    file_specific_output_dir = output_base_dir / parquet_file_output_dir_name
    file_specific_output_dir.mkdir(parents=True, exist_ok=True)


    logging.info(f"Found {len(dataset)} records. Showing details of random ones...")

    for i, record in enumerate(dataset):
        # if random.randint(0, 10) > 2:
        #     continue

        logging.info(f"\n  --- Record {i} in {parquet_path.name} ---")

        # 2. Print 'bbox'
        if 'bbox' in record and record['bbox'] is not None:
            logging.info(f"    bbox: {record['bbox']}")
            logging.info(f"    bbox type: {type(record['bbox'])}")
        elif 'bbox' in features:
            # logging.info(f"    bbox: (present but None/empty)")
            continue
        else:
            # logging.info(f"    bbox: (column not found in this record/features)")
            continue

        # 3. Save images
        if image_columns_info:
            logging.info("    Image data:")
            for img_col_name, img_type_enum in image_columns_info:
                pil_img = None
                try:
                    if img_type_enum == IMAGE_TYPE_HF:
                        img_data = record.get(img_col_name)
                        if img_data and isinstance(img_data, PILImage.Image): # Already a PIL Image
                            pil_img = img_data
                        elif img_data and 'bytes' in img_data and isinstance(img_data['bytes'], bytes): # Sometimes HFImage might be dict-like
                            pil_img = PILImage.open(io.BytesIO(img_data['bytes']))
                        elif img_data: # Should be PIL if HFImage
                             logging.warning(f"      Column {img_col_name} is HFImage type but content is not PIL.Image: {type(img_data)}")


                    elif img_type_enum == IMAGE_TYPE_BYTES_DICT:
                        img_dict = record.get(img_col_name)
                        if img_dict and 'bytes' in img_dict and isinstance(img_dict['bytes'], bytes):
                            pil_img = PILImage.open(io.BytesIO(img_dict['bytes']))
                        elif img_dict:
                            logging.warning(f"      Column {img_col_name} (bytes dict) has unexpected content for 'bytes': {type(img_dict.get('bytes'))}")

                    if pil_img:
                        # Sanitize image column name for filename
                        img_col_filename_part = "".join(c if c.isalnum() else '_' for c in img_col_name)
                        image_save_path = file_specific_output_dir / f"record_{i}_{img_col_filename_part}.png"

                        pil_img.save(image_save_path)
                        logging.info(f"      Saved image from column '{img_col_name}' to: {image_save_path}")

                        # if 'bbox' in record and record['bbox'] is not None:
                        #     # Optionally draw bbox on the image
                        #     json_response = json.loads(record['bbox'])
                        #     bboxes = json_response_to_bboxes(json_response)
                        #     draw_bboxes(pil_img, bboxes, SHOW_OBJECT_NAME=True)
                        #     bbox_image_save_path = file_specific_output_dir / f"record_{i}_{img_col_filename_part}_bbox.png"
                        #     pil_img.save(bbox_image_save_path)
                        # logging.info(f"      Image with bbox saved to: {bbox_image_save_path}")

                    elif record.get(img_col_name) is not None:
                        logging.warning(f"      Could not decode or access image data for column '{img_col_name}'.")
                    # else: image data is None, skip silently

                except Exception as e_img:
                    logging.error(f"      Error processing image column '{img_col_name}' for record {i}: {e_img}")
        else:
            logging.info("    No image columns identified to save.")

        action = user_prompt(f"View next record in {parquet_path.name}? (y/n/q to quit all, f for next file): ", default_yes=True)
        if action == "quit": return "quit"
        if action == False: break # Stop viewing records for this file, move to next file prompt
        if isinstance(action, str) and action.lower() == 'f': # Special command to skip to next file
            logging.info("Skipping to the next file...")
            return True # Continue to next file

    return True # Successfully processed this file (or user chose to skip records)

def main():
    parser = argparse.ArgumentParser(description="Explore LeRobot dataset Parquet files.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path of the LeRobot dataset directory (e.g., '~/lerobot/pusht'). This directory should contain the 'data' subdir or .parquet files directly.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_IMAGE_DIR),
        help=f"Directory to save extracted images. Default: {OUTPUT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--parquet-file",
        type=str,
        default=None,
        help="Path to a specific Parquet file to explore. If provided, will ignore --dataset-path and only explore this file.",   
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if output_dir.exists():
        logging.warning(f"Output directory {output_dir} already exists.")
        action = user_prompt(f"Clear and overwrite {output_dir}? (y/n/q): ")
        if action == "quit":
            logging.info("Exiting.")
            sys.exit(0)
        if action:
            try:
                shutil.rmtree(output_dir)
                logging.info(f"Removed existing output directory: {output_dir}")
            except Exception as e:
                logging.error(f"Could not remove existing output directory {output_dir}: {e}")
                sys.exit(1)
        else:
            logging.info(f"Proceeding with existing output directory: {output_dir}. Files may be overwritten or mixed.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Extracted images will be saved in: {output_dir}")

    if args.parquet_file:
        result = explore_parquet_file(args.parquet_file, output_dir)
    else:
        dataset_root = Path(args.dataset_path)
        parquet_file_paths = find_parquet_files(dataset_root)

        if not parquet_file_paths:
            logging.error(f"No Parquet files found in {dataset_root}. Exiting.")
            sys.exit(1)

        for i, parquet_file_str in enumerate(parquet_file_paths):
            logging.info(f"\n--- Processing file {i+1} of {len(parquet_file_paths)} ---")
            result = explore_parquet_file(parquet_file_str, output_dir)
            if result == "quit":
                logging.info("User requested to quit. Exiting.")
                break
            elif result == False: # Error in loading/processing file itself
                action = user_prompt(f"Error with file {Path(parquet_file_str).name}. Continue to next file? (y/n/q): ")
                if action == "quit" or not action:
                    logging.info("Exiting.")
                    break
            # If result is True, continue to the next file implicitly or after record loop ends.
            if i < len(parquet_file_paths) -1 and result != "quit": # Don't ask after last file
                action = user_prompt(f"Continue to the next Parquet file? (y/n/q): ")
                if action == "quit" or not action:
                    logging.info("Exiting.")
                    break
    logging.info("\nExploration finished.")

if __name__ == "__main__":
    main()