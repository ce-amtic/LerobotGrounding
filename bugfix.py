import pandas as pd
import json
from pathlib import Path
from datasets import load_dataset
from datasets.features.image import Image as ImageHF
from PIL import Image
from tqdm import tqdm
import io
from qwen_grounding_utils_v3 import improved_json_parser, normolize_bbox, load_list_from_jsonl

def process_parquet(parquet_path: str):
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        # print(f"[跳过] 文件不存在: {parquet_path}")
        return

    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as e:
        print(f"[错误] 无法读取 {parquet_path}：{e}")
        return

    if 'bbox' not in df.columns:
        # print(f"[跳过] 没有 'bbox' 列: {parquet_path}")
        return

    if 'observation.images.cam' not in df.columns:
        # print(f"[跳过] 没有 'observation.images.cam' 列: {parquet_path}")
        return

    updated = False
    for idx, row in df.iterrows():
        bbox_str = row['bbox']
        if not isinstance(bbox_str, str):
            continue

        try:
            bbox_dict = json.loads(bbox_str)
        except Exception:
            continue

        if 'error' not in bbox_dict or 'original_response' not in bbox_dict:
            continue

        text_response = bbox_dict['original_response']

        # 处理图像获取 W, H
        image_col_name = 'observation.images.cam'
        pil_image = None
        image_data = row[image_col_name]

        if isinstance(image_data, Image.Image):
            pil_image = image_data
        elif isinstance(image_data, dict):
            if image_data and image_data.get("bytes"):
                pil_image = Image.open(io.BytesIO(image_data["bytes"]))

        if pil_image is None:
            # print(f"[跳过] 无法读取图像: 行号 {idx}")
            continue

        W, H = pil_image.size

        # 替换 bbox 字段
        try:
            raw_json_response = improved_json_parser(text_response)
            json_response = normolize_bbox(raw_json_response, W, H)
            df.at[idx, 'bbox'] = json.dumps(json_response, ensure_ascii=False)
            updated = True
        except Exception as e:
            print(f"[警告] 解析失败 {parquet_path} 行号 {idx}: {e}")
            continue

    if updated:
        # backup_path = parquet_path.with_suffix(".bak.parquet")
        # parquet_path.rename(backup_path)
        df.to_parquet(parquet_path, engine="pyarrow")
        print(f"[完成] 修复并覆盖保存: {parquet_path}")
    else:
        print(f"[完成] 无需修改: {parquet_path}")


parquet_list = load_list_from_jsonl("bugfix.jsonl")
parquet_list = [item["path"] for item in parquet_list]
for parquet_path in tqdm(parquet_list, desc="Processing Parquet Files"):
    process_parquet(parquet_path)