import json, sys, logging, random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from qwen_grounding_utils_v3 import load_list_from_jsonl

dataset_info_list = [{'num': 50, 'path': '/pdata/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds'}, {'num': 51580, 'path': '/pdata/oxe_lerobot/bc_z'}, {'num': 415, 'path': '/pdata/oxe_lerobot/berkeley_fanuc_manipulation'}, {'num': 435, 'path': '/pdata/oxe_lerobot/nyu_door_opening_surprising_effectiveness'}, {'num': 18250, 'path': '/pdata/oxe_lerobot/robo_set'}, {'num': 1355, 'path': '/pdata/oxe_lerobot/ucsd_pick_and_place_dataset_converted_externally_to_rlds'}, {'num': 570, 'path': '/pdata/oxe_lerobot/stanford_hydra_dataset_converted_externally_to_rlds'}, {'num': 77960, 'path': '/pdata/oxe_lerobot/droid'}, {'num': 522, 'path': '/pdata/oxe_lerobot/iamlab_cmu_pickup_insert_converted_externally_to_rlds'}, {'num': 3847, 'path': '/pdata/oxe_lerobot/io_ai_tech'}, {'num': 576, 'path': '/pdata/oxe_lerobot/cmu_play_fusion'}, {'num': 1141, 'path': '/pdata/oxe_lerobot/roboturk'}, {'num': 8611, 'path': '/pdata/oxe_lerobot/fmb'}, {'num': 69757, 'path': '/pdata/oxe_lerobot/language_table'}, {'num': 480, 'path': '/pdata/oxe_lerobot/berkeley_mvp_converted_externally_to_rlds'}, {'num': 2460, 'path': '/pdata/oxe_lerobot/stanford_robocook_converted_externally_to_rlds'}, {'num': 107, 'path': '/pdata/oxe_lerobot/dlr_sara_grid_clamp_converted_externally_to_rlds'}, {'num': 402, 'path': '/pdata/oxe_lerobot/plex_robosuite'}, {'num': 100, 'path': '/pdata/oxe_lerobot/dlr_sara_pour_converted_externally_to_rlds'}, {'num': 9929, 'path': '/pdata/oxe_lerobot/vima_converted_externally_to_rlds'}, {'num': 240, 'path': '/pdata/oxe_lerobot/austin_sailor_dataset_converted_externally_to_rlds'}, {'num': 6484, 'path': '/pdata/oxe_lerobot/taco_play'}, {'num': 9109, 'path': '/pdata/oxe_lerobot/stanford_mask_vit_converted_externally_to_rlds'}, {'num': 1442, 'path': '/pdata/oxe_lerobot/berkeley_cable_routing'}, {'num': 53177, 'path': '/pdata/oxe_lerobot/bridge_data_v2'}, {'num': 896, 'path': '/pdata/oxe_lerobot/berkeley_autolab_ur5'}, {'num': 25460, 'path': '/pdata/oxe_lerobot/bridge'}, {'num': 559, 'path': '/pdata/oxe_lerobot/austin_sirius_dataset_converted_externally_to_rlds'}, {'num': 816606, 'path': '/pdata/oxe_lerobot/mt_opt_rlds'}, {'num': 5208, 'path': '/pdata/oxe_lerobot/dobbe'}, {'num': 745, 'path': '/pdata/oxe_lerobot/viola'}, {'num': 135, 'path': '/pdata/oxe_lerobot/cmu_stretch'}, {'num': 1500, 'path': '/pdata/oxe_lerobot/utaustin_mutex'}, {'num': 902, 'path': '/pdata/oxe_lerobot/toto'}, {'num': 87212, 'path': '/pdata/oxe_lerobot/fractal20220817_data'}, {'num': 976, 'path': '/pdata/oxe_lerobot/jaco_play'}, {'num': 82775, 'path': '/pdata/oxe_lerobot/robo_net'}, {'num': 580392, 'path': '/pdata/oxe_lerobot/kuka'}, {'num': 150, 'path': '/pdata/oxe_lerobot/ucsd_kitchen_dataset_converted_externally_to_rlds'}, {'num': 200, 'path': '/pdata/oxe_lerobot/qut_dexterous_manpulation'}, {'num': 365, 'path': '/pdata/oxe_lerobot/nyu_franka_play_dataset_converted_externally_to_rlds'}, {'num': 122, 'path': '/pdata/oxe_lerobot/columbia_cairlab_pusht_real'}, {'num': 13, 'path': '/pdata/oxe_lerobot/uiuc_d3field'}, {'num': 104, 'path': '/pdata/oxe_lerobot/dlr_edan_shared_control_converted_externally_to_rlds'}]
dataset_size = {item['path']: item['num'] for item in dataset_info_list}

def load_finished_list(world_size):
    finished_list = []
    for i in range(world_size):
        finished_file = Path(f"resume/finished_parquets_{i}.jsonl")
        if finished_file.exists():
            part_list = load_list_from_jsonl(finished_file)
            part_list = [item["path"] for item in part_list]
            finished_list.extend(part_list)
    return finished_list

def load_parquet(dataset_path, max_files=None):
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
        # parquet_files = [str(f) for f in chunk_folder.iterdir() if f.is_file() and f.name.endswith(".parquet")]
        parquet_files = []
        for f in chunk_folder.iterdir():
            if f.is_file() and f.name.endswith(".parquet"):
                parquet_files.append(str(f))
            if max_files is not None:
                if len(parquet_files) >= max_files:
                    break
        if not parquet_files:
            logging.error(f"No parquet files found in {chunk_folder}.")
            continue
        parquet_paths.extend(parquet_files)
        if max_files is not None and len(parquet_paths) >= max_files:
            break
    
    # logging.info(f"Found {len(parquet_paths)} parquet files in {dataset_path}.")
    if max_files is not None:
        return parquet_paths[:max_files]
    return parquet_paths

def check_dataset_finished(dataset_path, dataset_size, finished_list):
    all_count = dataset_size.get(dataset_path, 0)

    if not dataset_path.endswith('/'):
        dataset_path += '/'
    finished_list = [x for x in finished_list if x.startswith(dataset_path)]
    finished_count = len(finished_list)

    finish_rate = finished_count / all_count
    print(f"Dataset: {dataset_path}, Finished Count: {finished_count}, Total Count: {all_count}, Finish Rate: {finish_rate:.2f}")
    return finish_rate >= 0.8
    # all_parquet_files = load_parquet(dataset_path)
    # all_count = len(all_parquet_files)
    # nope_count = 0
    # for parquet_path_str in tqdm(all_parquet_files, desc="Checking Parquet Files", unit="file"):
    #     if parquet_path_str not in finished_list:
    #         nope_count += 1
            # df = pd.read_parquet(parquet_path_str)
            # if 'sub_task_index' not in df.columns:
            #     continue
            # print(f"Found unfinished parquet file: {parquet_path_str}")
            # return False
            # return False
    # return True
    # finish_rate = (all_count - nope_count) / all_count
    # return finish_rate >= 0.8

def check_subtask_index(dataset_path):
    # size_limit = int(0.5 * dataset_size[dataset_path])
    all_parquet_files = load_parquet(dataset_path)
    for parquet_path_str in tqdm(all_parquet_files, desc="Checking Parquet Files", unit="file"):
        try:
            df = pd.read_parquet(parquet_path_str)
        except Exception as e:
            # logging.error(f"Error reading {parquet_path_str}: {e}")
            continue
        if 'sub_task_index' in df.columns:
            # logging.error(f"Missing 'sub_task_index' in {parquet_path_str}")
            return True
    return False

if __name__ == "__main__":
    dataset_home = '/pdata/oxe_lerobot'
    all_datasets = [str(f) for f in Path(dataset_home).iterdir() if f.is_dir()]

    subtask_failed_datasets = []
    final_datasets = []
    for dataset_path in all_datasets:
        if not check_subtask_index(dataset_path):
            print(f"Negative sub_task_index: {dataset_path}")
            subtask_failed_datasets.append(dataset_path)
            continue
        final_datasets.append(dataset_path)
    
    if subtask_failed_datasets:
        print(f"Subtask index failed datasets: {len(subtask_failed_datasets)}\n{subtask_failed_datasets}")
    if final_datasets:
        print(f"Final datasets with sub_task_index: {len(final_datasets)}\n{final_datasets}")