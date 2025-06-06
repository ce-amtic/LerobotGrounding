import math, random
import torch
import os, re
import json, ast
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple, Dict, Any # Added for type hinting
import tempfile # For saving PIL images to pass paths for vLLM

# Attempt to import vLLM and qwen_vl_utils components
# These are necessary for the vLLM-based inference
_VLLM_IMPORTED = False
_LLM_CLASS = None
_SAMPLING_PARAMS_CLASS = None
try:
    from vllm import LLM, SamplingParams
    _LLM_CLASS = LLM
    _SAMPLING_PARAMS_CLASS = SamplingParams
    _VLLM_IMPORTED = True
except ImportError:
    print("WARNING: vLLM not found. Please install vLLM (e.g., `pip install vllm`) to use vLLM-based inference.")

_PROCESS_VISION_INFO_DEFAULT_IMPORTED = False
_PVI_FUNCTION = None
try:
    from qwen_vl_utils import process_vision_info
    _PVI_FUNCTION = process_vision_info
    _PROCESS_VISION_INFO_DEFAULT_IMPORTED = True
except ImportError:
    print("WARNING: qwen_vl_utils.process_vision_info not found. "
          "This function is required for vLLM inference with Qwen-VL. "
          "Ensure qwen_vl_utils.py is in your PYTHONPATH or provide the function via 'process_vision_info_func' argument.")

# Transformers AutoProcessor is still needed
_TRANSFORMERS_PROCESSOR_IMPORTED = False
_AUTOPROCESSOR_CLASS = None
try:
    from transformers import AutoProcessor
    _AUTOPROCESSOR_CLASS = AutoProcessor
    _TRANSFORMERS_PROCESSOR_IMPORTED = True
except ImportError:
    print("WARNING: Transformers library not found. Please install transformers (e.g., `pip install transformers`)")


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

def normolize_bbox(json_response, width, height):
    processed_json_response = []
    for obj in json_response:
        bbox = obj.get('bbox_2d')
        label = obj.get('label')
        if not bbox:
            continue
        bbox = move_to_cpu(bbox)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        if not (x1 < x2 and y1 < y2): continue

        # normolize to [0, 1]
        x1 /= width
        y1 /= height
        x2 /= width
        y2 /= height
        processed_json_response.append({
            'bbox_2d': [x1, y1, x2, y2],
            'label': label
        })
    return processed_json_response


### --- model infer --- ###
def setup_model(MODEL_NAME: str, tensor_parallel_size: int = 1, trust_remote_code: bool = True, **vllm_kwargs: Any):
    """
    Sets up the vLLM engine and a Hugging Face AutoProcessor.
    """
    if not _VLLM_IMPORTED:
        raise ImportError("vLLM library is not installed. Please install it to use this function.")
    if not _TRANSFORMERS_PROCESSOR_IMPORTED:
        raise ImportError("Transformers library is not installed. Please install it.")

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("CUDA is not available or no GPUs detected. vLLM requires CUDA-enabled GPUs.")
        raise RuntimeError("vLLM requires CUDA. No GPU found or CUDA not available.")
    
    num_available_gpus = torch.cuda.device_count()
    print(f"CUDA available. Number of GPUs: {num_available_gpus}")
    if tensor_parallel_size > num_available_gpus:
        print(f"Warning: tensor_parallel_size ({tensor_parallel_size}) > available GPUs ({num_available_gpus}). "
              f"Adjusting tensor_parallel_size to {num_available_gpus}.")
        tensor_parallel_size = num_available_gpus

    print(f"Initializing AutoProcessor for: {MODEL_NAME}")
    processor = _AUTOPROCESSOR_CLASS.from_pretrained(MODEL_NAME, trust_remote_code=trust_remote_code)

    print(f"Initializing vLLM engine for model: {MODEL_NAME} with tensor_parallel_size: {tensor_parallel_size}")
    
    # Default vLLM keyword arguments
    # limit_mm_per_prompt: max number of multimodal inputs (images/videos) per prompt string.
    # Since we construct one prompt string per image, 1 is appropriate here.
    default_llm_params = {
        "limit_mm_per_prompt": {"image": 1, "video": 0}, # Max 1 image, 0 videos per prompt
        "trust_remote_code": trust_remote_code,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": "auto"  # vLLM handles dtype selection (e.g., bfloat16, float16)
    }
    # User-provided vllm_kwargs will override defaults
    final_llm_params = {**default_llm_params, **vllm_kwargs}
    
    llm = _LLM_CLASS(model=MODEL_NAME, **final_llm_params)
    return llm, processor


def batched_inference(
    images: List[Image.Image],
    prompts: List[str],
    llm: 'LLM', # vLLM engine instance from setup_model
    processor: 'AutoProcessor', # AutoProcessor instance from setup_model
    system_prompt: str = "You are a helpful assistant",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.001,
    repetition_penalty: float = 1.05,
    stop_token_ids: Optional[List[int]] = None,
    process_vision_info_func: Optional[callable] = None,
    temp_dir_for_images: Optional[str] = None # Optional: specify a directory for temp image files
) -> Tuple[List[str], List[int], List[int]]:
    """
    Performs batched inference using vLLM for Qwen-VL style models.
    """
    if not _VLLM_IMPORTED:
        raise ImportError("vLLM library is not installed. Please install it to use this function.")
    if llm is None or processor is None: # llm is the vLLM engine
        raise ValueError("Please provide vLLM engine and processor (e.g., from setup_model()).")

    # Resolve the process_vision_info function
    pvi_func = process_vision_info_func
    if pvi_func is None:
        if _PROCESS_VISION_INFO_DEFAULT_IMPORTED:
            pvi_func = _PVI_FUNCTION
        else:
            raise RuntimeError(
                "qwen_vl_utils.process_vision_info is not available by default import. "
                "Please ensure qwen_vl_utils.py is in your PYTHONPATH or "
                "pass a valid function to 'process_vision_info_func'."
            )

    if not images or not prompts:
        return [], [], []
    if len(images) != len(prompts):
        raise ValueError("Number of images and prompts must match for batching.")

    # Prepare sampling parameters for vLLM
    sampling_params_obj = _SAMPLING_PARAMS_CLASS(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids if stop_token_ids is not None else []
    )

    llm_inputs_batch = []
    # temp_image_files = [] # To keep track of temporary files for cleanup

    for i, (image_pil, prompt_text) in enumerate(zip(images, prompts)):

        # Construct messages in the format expected by Qwen-VL processor and process_vision_info
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image",
                        "image": image_pil,
                        # "image": temp_image_path, # Pass the path to the temporary image file
                        # Optional: min_pixels/max_pixels if your process_vision_info requires/supports them
                        # "min_pixels": 224 * 224,
                        # "max_pixels": 1280 * 28 * 28, # Example values
                    },
                ],
            },
        ]

        # Apply chat template to get the full prompt string
        prompt_str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Use the provided process_vision_info function
        # This function is expected to process messages and extract multimodal data
        image_inputs, video_inputs, video_kwargs = pvi_func(messages, return_video_kwargs=True)
            
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        # video_inputs are not expected in this image-only pipeline
        # if video_inputs is not None: mm_data["video"] = video_inputs 
            
        current_llm_input = {
            "prompt": prompt_str,
            "multi_modal_data": mm_data,
        }
        # Add mm_processor_kwargs if video_kwargs are returned (as per vLLM example)
        if video_kwargs:
            current_llm_input["mm_processor_kwargs"] = video_kwargs
            
        llm_inputs_batch.append(current_llm_input)

    # Perform batched inference with vLLM
    # llm.generate can take a list of these structured inputs
    vllm_outputs = llm.generate(llm_inputs_batch, sampling_params_obj)
        
    output_texts = [output.outputs[0].text for output in vllm_outputs]
        
    # Use original image dimensions. If process_vision_info provides processed dimensions,
    # that would be more accurate for tasks like bounding box scaling, but original
    # dimensions are a reasonable default.
    input_heights = [img.height for img in images]
    input_widths = [img.width for img in images]

    return output_texts, input_heights, input_widths

    # finally:
    #     # Clean up temporary image files
    #     for f_path in temp_image_files:
    #         try:
    #             if os.path.exists(f_path):
    #                 os.remove(f_path)
    #         except OSError as e:
    #             print(f"Warning: Could not delete temporary file {f_path}. Error: {e}")


def grounding_pipeline_batched(
    images: List[Image.Image],
    model=None, # This will be the vLLM engine instance
    processor=None, # This will be the AutoProcessor instance
    task_descs: Optional[List[Optional[str]]] = None,
    SHOW_OBJECT_NAME: bool = False, # For draw_bboxes, if used
    USE_SUBTASK_CONDITIONING: bool = False
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]: # Return type for drawn images might be None
    if not images:
        return [], [] # Or ([], None)

    if model is None or processor is None:
        raise ValueError(
            "Please provide the vLLM engine and processor, e.g., from setup_model(). "
            "The 'model' argument should be the vLLM LLM instance."
        )

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

    # Call the vLLM-backed batched_inference
    # Default system_prompt, max_new_tokens, etc., from batched_inference signature will be used
    # if not specified here.
    all_text_responses, _, _ = batched_inference(
        images,
        questions,
        llm=model,  # Pass the vLLM engine instance as 'llm'
        processor=processor,
        # system_prompt and sampling parameters can be customized here if needed:
        # system_prompt="Your custom system prompt",
        # max_new_tokens=512,
        # temperature=0.2,
    )

    all_json_responses = []
    # The original code snippet for qwen_grounding_utils_v2.py has image drawing utilities
    # (parse_bboxes, draw_bboxes) commented out, and this function returns None for images.
    # This behavior is maintained. If image drawing is re-enabled, all_input_heights/widths
    # from batched_inference might be useful for parse_bboxes.

    for i in range(len(images)):
        text_response = all_text_responses[i]
        H, W = images[i].height, images[i].width
        try:
            raw_json_response = improved_json_parser(text_response)
            json_response = normolize_bbox(raw_json_response, W, H)
        except Exception as e:
            # print(f"Failed to parse model output for image {i}: {text_response}. Error: {e}")
            json_response = {"error": f"Failed to parse JSON: {str(e)}", "original_response": text_response}
        all_json_responses.append(json_response)

    return all_json_responses, None # Returning None for images as per the provided snippet


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
# The vLLM example previously in this section has been incorporated into the new functions.
# Original Hugging Face Transformers based setup_model and batched_inference are replaced by vLLM versions.