import math, random
import torch
import os, re
import json, ast
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import List, Optional, Tuple, Dict, Any # Added for type hinting

### --- process images --- ###
def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().item() # .item() if it's a 0-dim tensor, otherwise .tolist() or just keep as tensor
    elif isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    else:
        return obj

def improved_json_parser(text_response: str):
    if not isinstance(text_response, str):
        raise TypeError("Input must be a string.")
    text_response = text_response.strip()
    try:
        return json.loads(text_response)
    except json.JSONDecodeError:
        pass
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```"
    ]
    for pattern in patterns:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            potential_json = match.group(1).strip()
            if (potential_json.startswith('{') and potential_json.endswith('}')) or \
               (potential_json.startswith('[') and potential_json.endswith(']')):
                try:
                    return json.loads(potential_json)
                except json.JSONDecodeError:
                    pass
    first_bracket_idx = text_response.find('[')
    first_curly_idx = text_response.find('{')
    start_idx = -1
    expected_end_char = ''
    if first_bracket_idx != -1 and (first_curly_idx == -1 or first_bracket_idx < first_curly_idx):
        start_idx = first_bracket_idx
        expected_end_char = ']'
    elif first_curly_idx != -1:
        start_idx = first_curly_idx
        expected_end_char = '}'
    if start_idx != -1:
        end_idx = text_response.rfind(expected_end_char)
        if end_idx > start_idx:
            potential_json = text_response[start_idx : end_idx + 1]
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not extract valid JSON from the input string: '{text_response[:100]}...'")

def parse_bboxes(json_output, orig_width, orig_height):
    colors = ["red", "lime", "blue", "yellow", "cyan", "magenta", "deeppink", "orange", "purple", "lightgreen"]
    random.shuffle(colors)
    orig_height_cpu = move_to_cpu(orig_height) # Ensure it's a CPU value
    orig_width_cpu = move_to_cpu(orig_width)   # Ensure it's a CPU value
    bboxes = []
    if not isinstance(json_output, list): # Handle cases where json_output might not be a list of objects
        print(f"Warning: Expected a list of objects from JSON, but got {type(json_output)}. Full response: {json_output}")
        return []
    for i, obj in enumerate(json_output):
        color = colors[i % len(colors)]
        bbox = obj.get('bbox_2d')
        if not bbox:
            print(f"Skipping object {i} with no bbox_2d.")
            continue
        obj_name = obj.get('label', 'unknown')
        bbox = move_to_cpu(bbox)
        if not isinstance(bbox, list) or len(bbox) != 4:
            print(f"Invalid bbox format for object {i}: {bbox}. Expected a list of 4 values.")
            continue
        # Clamp bbox to image dimensions
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, orig_width_cpu))
        y1 = max(0, min(y1, orig_height_cpu))
        x2 = max(0, min(x2, orig_width_cpu))
        y2 = max(0, min(y2, orig_height_cpu))

        if not (x1 < x2 and y1 < y2):
            # print(f"Invalid bbox coordinates (x1>=x2 or y1>=y2) for object {i}: {[x1,y1,x2,y2]}. Original: {bbox}. Image bounds: {orig_width_cpu}x{orig_height_cpu}. Skipping.")
            # If one coordinate is at the boundary, it might be fine if it's a thin sliver. The issue is if x1>=x2 or y1>=y2.
            # A common case is if x1=x2 or y1=y2 due to being at the exact boundary.
            # Let's allow zero-width/height boxes if they are on the boundary, but not inverted ones.
            if x1 > x2 or y1 > y2:
                print(f"Invalid bbox coordinates (inverted) for object {i}: {[x1,y1,x2,y2]}. Original: {bbox}. Skipping.")
                continue
            # If x1==x2 or y1==y2, it's a line or point, technically not a box. We might still want to keep it.
            # For now, we'll keep it simple and require x1 < x2 and y1 < y2 for a drawable box.
            # If the model genuinely outputs such boxes, it might indicate an issue or a very specific edge case.
            # For robustness, we could skip or try to slightly adjust it.
            # print(f"Warning: Degenerate bbox for object {i}: {[x1,y1,x2,y2]}. (x1==x2 or y1==y2)")
            # Let's be strict for now.
            if not (x1 < x2 and y1 < y2):
                 print(f"Degenerate or invalid bbox for object {i}: {[x1,y1,x2,y2]} after clamping. Original: {bbox}. Skipping.")
                 continue


        bboxes.append({
            "box": (x1, y1, x2, y2),
            "name": obj_name,
            "color": color
        })
    if not bboxes:
        print(f"No valid bounding boxes found in the response.")
    return bboxes

def draw_bboxes(image, bboxes, SHOW_OBJECT_NAME=False):
    draw = ImageDraw.Draw(image)
    try:
        font_path = "fonts/CourierPrime-Regular.ttf"
        if not os.path.exists(font_path): # Fallback if specific font not found
            font_path = "arial.ttf" # Common system font, or remove for ImageFont.load_default()
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        font = ImageFont.load_default()
        print("Drawing text with default font")
    for bbox_info in bboxes:
        box, name, color = bbox_info["box"], bbox_info["name"], bbox_info["color"]
        draw.rectangle(box, outline=color, width=3)
        if SHOW_OBJECT_NAME:
            text_position = (box[0] + 2, box[1] - 18 if box[1] > 18 else box[1] + 2)
            draw.text(text_position, name, fill=color, font=font)
    return image



### --- model infer --- ###
def setup_model(MODEL_NAME):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("CUDA is not available or no GPUs detected. Using CPU.")
        print("WARNING: Qwen2.5-VL is large and will be very slow on CPU and may run out of memory.")
        device_map_arg = "cpu"
        model_dtype = torch.float32
        attn_implementation = None
    else:
        print(f"CUDA available. Number of GPUs: {torch.cuda.device_count()}")
        device_map_arg = "auto"
        model_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
        # attn_implementation = None # Use this if flash_attention_2 causes issues
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
        device_map=device_map_arg,
        attn_implementation=attn_implementation if attn_implementation else None
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    return model, processor

def batched_inference(
    images: List[Image.Image],
    prompts: List[str],
    system_prompt: str = "You are a helpful assistant",
    max_new_tokens: int = 1024, # Consider reducing this if JSON is typically shorter
    model=None,
    processor=None
) -> Tuple[List[str], List[int], List[int]]:
    if model is None or processor is None:
        raise ValueError("Please setup model and processor first using setup_model(), and pass them as arguments.")
    if not images or not prompts:
        return [], [], []
    if len(images) != len(prompts):
        raise ValueError("Number of images and prompts must match for batching.")

    batched_messages = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"image": "robot_operation_task.jpg"}]}
        ]
        batched_messages.append(messages)

    # The processor needs text in one list and images in another, corresponding by index
    texts_for_processor = [processor.apply_chat_template(msg_set, tokenize=False, add_generation_prompt=True) for msg_set in batched_messages]
    
    # Ensure all images are on CPU before passing to processor, if they aren't already PIL Images
    # PIL Images are fine. If they were tensors, they'd need to be on CPU.
    
    inputs = processor(text=texts_for_processor, images=images, padding=True, return_tensors="pt").to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Correctly slice generated IDs for batch
    generated_ids_batch = []
    for i in range(len(images)):
        input_len = inputs.input_ids[i].shape[0]
        generated_ids_batch.append(output_ids[i, input_len:])
        
    output_texts = processor.batch_decode(generated_ids_batch, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # image_grid_thw gives (num_patches_total, H_patches, W_patches) for each image in batch
    # input_heights = [inputs['image_grid_thw'][i][1].item() * 14 for i in range(len(images))]
    # input_widths = [inputs['image_grid_thw'][i][2].item() * 14 for i in range(len(images))]
    # The above is for the *patch grid*. For original image dimensions to processor:
    # Let's assume the processor handles resizing and the model output bboxes are relative to the *processed* image size.
    # The critical part is that parse_bboxes needs the dimensions *as seen by the model* for scaling.
    # The `inputs['image_grid_thw']` seems to provide info about how the image was patchified.
    # The height and width derived from `image_grid_thw` (e.g., `H_patches * 14`, `W_patches * 14`)
    # should correspond to the dimensions of the image *after* it has been processed by `processor`
    # and before being fed into the vision tower. This is what we need for `parse_bboxes`.
    
    input_heights = []
    input_widths = []
    # `image_grid_thw` is a tensor of shape (batch_size, num_total_patches, H_patches, W_patches)
    # No, documentation indicates it's a list of tensors, or if padded, a single tensor.
    # Let's check `inputs['image_sizes']` if available, or fallback to `image_grid_thw`
    # Typically, `inputs['pixel_values'].shape` would be (batch, channels, height, width) after processor
    
    if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
        # This seems specific to Qwen's older style or a particular configuration
        # It might be a list if images have different patch counts, or a tensor if padded.
        # Assuming it's a tensor of shape (batch_size, num_total_patches, H_patches, W_patches)
        # or a list of (num_total_patches, H_patches, W_patches)
        # The actual image size fed to ViT is (H_patches * patch_size, W_patches * patch_size)
        patch_size = 14 # Common for ViT-based models, check Qwen specifics if needed
        for i in range(len(images)):
            # If inputs['image_grid_thw'] is a single tensor (batch_size, total_patches, h_patches, w_patches)
            # Or if it's a list of tensors, one for each image in the batch
            grid_info = inputs['image_grid_thw'][i] # Get info for the i-th image
            h_patches = grid_info[1].item() # Assuming index 1 is H_patches
            w_patches = grid_info[2].item() # Assuming index 2 is W_patches
            input_heights.append(h_patches * patch_size)
            input_widths.append(w_patches * patch_size)
    else:
        # Fallback: use original image sizes. This might lead to incorrect bbox scaling
        # if the model internally resizes and outputs bboxes relative to the resized image.
        # This is a critical point: bboxes from model are usually normalized or relative to model's input view.
        print("Warning: Could not determine processed image dimensions from model inputs. Using original image sizes for bbox parsing. This might be inaccurate.")
        input_heights = [img.height for img in images]
        input_widths = [img.width for img in images]

    return output_texts, input_heights, input_widths


def grounding_pipeline_batched(
    images: List[Image.Image],
    model=None,
    processor=None,
    task_descs: Optional[List[Optional[str]]] = None,
    SHOW_OBJECT_NAME: bool = False,
    USE_SUBTASK_CONDITIONING: bool = False
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    if not images:
        return [], []

    if task_descs is None:
        task_descs = [None] * len(images)
    
    if len(images) != len(task_descs):
        raise ValueError("Number of images and task_descs must match.")

    questions = []
    for i, task_desc in enumerate(task_descs):
        if USE_SUBTASK_CONDITIONING and task_desc:
            question = (
                f"This is a picture of using a robotic arm to complete a specific task. The task completed in the picture is: {task_desc}.\n"
                "Box out the items relevant with the task in the image, output its bbox coordinates using JSON format."
            )
        else:
            question = (
                "This is a picture of using a robotic arm to complete a specific task.\n"
                "Box out the items relevant with the task in the image, output its bbox coordinates using JSON format."
            )
        questions.append(question)

    all_text_responses, all_input_heights, all_input_widths = batched_inference(
        images, questions, model=model, processor=processor
    )

    all_json_responses = []
    all_output_images = []

    for i in range(len(images)):
        current_image = images[i]
        text_response = all_text_responses[i]
        input_height = all_input_heights[i]
        input_width = all_input_widths[i]
        
        try:
            json_response = improved_json_parser(text_response)
        except Exception as e:
            print(f"Failed to parse model output for image {i}: {text_response}. Error: {e}")
            json_response = {} # Empty dict for failed parsing
            all_json_responses.append(json_response)
            all_output_images.append(current_image.copy()) # Return original image on error
            continue

        # parse_bboxes needs width and height of the image *as seen by the model*
        # to correctly interpret (potentially normalized or relative) bbox coordinates.
        bboxes = parse_bboxes(json_response, input_width, input_height)
        
        output_image_copy = current_image.copy() # Work on a copy
        if bboxes:
            output_image_processed = draw_bboxes(output_image_copy, bboxes, SHOW_OBJECT_NAME)
        else:
            # print(f"No valid bounding boxes found in the response for image {i}. Returning original image.")
            output_image_processed = output_image_copy # Already a copy

        all_json_responses.append(json_response) # Store raw JSON from model
        all_output_images.append(output_image_processed)

    return all_json_responses, all_output_images



### --- helpers --- ###
def load_list_from_jsonl(jsonl_path):
    data_list = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    return data_list
def save_list_to_jsonl(data_list, jsonl_path):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # df = pandas.DataFrame(bboxes)
    # df.to_json(bboxes_json_path, orient="records", lines=True, force_ascii=False)




### --- legacy --- ###