import math, random
import torch
import os, re
import json, ast
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

### --- process images --- ###
def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().item()
    elif isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    else:
        return obj

def improved_json_parser(text_response: str):
    """
    Tries to parse a JSON string that might be embedded in other text
    or Markdown code blocks.
    """
    if not isinstance(text_response, str):
        raise TypeError("Input must be a string.")

    text_response = text_response.strip() # 去除首尾空白

    # 1. 尝试直接解析 (如果已经是纯净的 JSON)
    try:
        return json.loads(text_response)
    except json.JSONDecodeError:
        pass # 继续尝试其他方法

    # 2. 尝试从 Markdown 代码块中提取 JSON
    #    - ```json ... ```
    #    - ``` ... ```
    # re.DOTALL (or re.S) makes '.' match newlines as well
    patterns = [
        r"```json\s*(.*?)\s*```",  # 匹配 ```json ... ```
        r"```\s*(.*?)\s*```"      # 匹配 ``` ... ```
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

    # 3. 尝试查找第一个 '{' 或 '[' 到最后一个 '}' 或 ']' 之间的内容
    #    这是一种更通用的方法，但也可能因为字符串中其他地方的括号而出错
    #    我们假设我们想要的是主要的 JSON 对象/数组
    first_bracket_idx = text_response.find('[')
    first_curly_idx = text_response.find('{')

    start_idx = -1
    expected_end_char = ''

    # 确定 JSON 是以 '[' (数组) 还是 '{' (对象) 开始
    if first_bracket_idx != -1 and (first_curly_idx == -1 or first_bracket_idx < first_curly_idx):
        start_idx = first_bracket_idx
        expected_end_char = ']'
    elif first_curly_idx != -1:
        start_idx = first_curly_idx
        expected_end_char = '}'
    
    if start_idx != -1:
        # 查找对应的最后一个结束括号
        # 注意: 这仍然是一个启发式方法，对于嵌套且外部有干扰括号的情况可能不完美
        # 但对于常见的 LLM 输出包裹的 JSON 应该能工作
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

    orig_height = move_to_cpu(orig_height)
    orig_width = move_to_cpu(orig_width)
    
    bboxes = []
    for i, obj in enumerate(json_output):
        color = colors[i % len(colors)]
        bbox = obj.get('bbox_2d')
        if not bbox:
            print(f"Skipping object {i} with no bbox_2d.")
            continue 
        obj_name = obj.get('label', 'unknown')

        # check data validation
        bbox = move_to_cpu(bbox)
        if not isinstance(bbox, list) or len(bbox) != 4:
            print(f"Invalid bbox format for object {i}: {bbox}. Expected a list of 4 values.")
            continue
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > orig_width or bbox[3] > orig_height:
            print(f"Invalid bbox coordinates for object {i}: {bbox}. Out of image bounds.")
            bbox = [
                max(0, bbox[0]), max(0, bbox[1]),
                min(orig_width, bbox[2]), min(orig_height, bbox[3])
            ]
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            print(f"Invalid bbox coordinates for object {i}: {bbox}. Expected (x1 < x2 and y1 < y2).")
            continue

        bbox = move_to_cpu(bbox)
        bboxes.append({
            "box": (bbox[0], bbox[1], bbox[2], bbox[3]),
            "name": obj_name,
            "color": color
        })

    if not bboxes:
        print(f"No valid bounding boxes found in the response.")
        return []
    return bboxes

def draw_bboxes(image, bboxes, SHOW_OBJECT_NAME=False):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("fonts/CourierPrime-Regular.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        print("drawing text with default font")

    for bbox_info in bboxes:
        box, name, color = bbox_info["box"], bbox_info["name"], bbox_info["color"]
        draw.rectangle(box, outline=color, width=3) # Increased width for visibility
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
        attn_implementation = None # No flash attention on CPU
        gpu_available = False
    else:
        print(f"CUDA available. Number of GPUs: {torch.cuda.device_count()}")
        device_map_arg = "auto" # Qwen's recommended way for multi-GPU
        model_dtype = torch.bfloat16 # Recommended for Qwen VL with Ampere+ GPUs
        # Enable flash_attention_2 if your environment supports it (typically Ampere+ GPUs and recent PyTorch/CUDA)
        attn_implementation = "flash_attention_2"
        # attn_implementation = None # Use this if flash_attention_2 causes issues
        gpu_available = True

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
        device_map=device_map_arg,
        attn_implementation=attn_implementation if attn_implementation else None # Pass None if not using
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    return model, processor


def inference(image, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024, model=None, processor=None):
    if model is None or processor is None:
        raise ValueError("Please setup model and processor first using setup_model(), and pass them as arguments.")

    messages = [
        {
        "role": "system",
        "content": system_prompt
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "image": "robot_operation_task.jpg"
            }
        ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14

    return output_text[0], input_height, input_width

def grounding_pipeline(image: Image, model=None, processor=None, task_desc=None, SHOW_OBJECT_NAME=False, USE_SUBTASK_CONDITIONING=False):
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

    text_response, input_height, input_width = inference(image, question, model=model, processor=processor)
    try:
        json_response = improved_json_parser(text_response)
    except Exception as e:
        print(f"Failed to parse model output: {text_response}")
        return {}, image.copy()

    bboxes = parse_bboxes(json_response, input_width, input_height)
    if bboxes:
        output_image = draw_bboxes(image, bboxes, SHOW_OBJECT_NAME)
    else:
        print("No valid bounding boxes found in the response. Returning original image.")
        output_image = image.copy()

    return json_response, output_image