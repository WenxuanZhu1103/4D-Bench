import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import os
import json
import numpy as np
from PIL import Image
import copy
import warnings
from decord import VideoReader, cpu
import sys
import re

warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num):
    if isinstance(video_path, str):
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def submit_vqa_to_model(model, tokenizer, image_processor, vqa, frames, device="cuda"):
    vqa = f"You are a excellent video analyst. I provide you 18 frames with every six images uniformly sampled from one video, each video captured from a different angle and a question and four choices. \
                        Carefully watch the provided videos and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question. Here is the question and choices: \n {vqa}. \n \
                        You must return only the option identifier (e.g., '(A)') without any additional text, do not add any additional analysis, just return the correct option identifier."                       
    
    image_tensors = []
    processed_frames = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(processed_frames)
    
    conv_template = "qwen_1_5"
    question = f"{DEFAULT_IMAGE_TOKEN}\n{vqa}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.shape[:2] for frame in frames]
    
    outputs = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(response)
    return response

def extract_answer_option(text):

    paren_pattern = r'\(([A-D])\)'
    matches = re.findall(paren_pattern, text)
    if matches:
        return matches[0]
    
    isolated_pattern = r'(?:^|[\s\(\.,;:])([A-D])(?:[\s\)\.,;:]|$)'
    matches = re.findall(isolated_pattern, text)
    if matches:
        return matches[0]
    
    return None

def handle_vqa_result(pred_answer, correct_answer_index):

    if pred_answer is None or not isinstance(pred_answer, str):
        print("Invalid input: prediction answer is empty or not a string")
        return -2, -1

    extracted_option = extract_answer_option(pred_answer)
    if not extracted_option:
        print("Invalid response format. No valid option (A/B/C/D) found")
        return -1, -1

    letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    pred_number = letter_to_number[extracted_option]
    
    if pred_number == correct_answer_index:
        print("Correct answer")
        return 1, pred_number
    else:
        print("Wrong answer")
        return 0, pred_number
    
def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='VQA Evaluation Script')
    parser.add_argument('--save_path', required=True, help='Path to save results')
    parser.add_argument('--vqa_file_path', required=True, help='Path to VQA data file')
    parser.add_argument('--video_data_path', required=True, help='Path to video data directory')
    parser.add_argument('--cache_dir', required=True, help='Path to cache directory for model weights')
    args = parser.parse_args()

    # Model initialization
    os.makedirs(args.save_path, exist_ok=True)
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map, 
        attn_implementation="sdpa",
        cache_dir=args.cache_dir
    )
    model.eval()

    # Use command line arguments for paths
    save_file_path = os.path.join(args.save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")

    # Load existing results if any
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}

    # Load VQA data
    with open(args.vqa_file_path, 'r') as f:
        vqa_data = json.load(f)

    view_ids = [1, 8, 16]

    for vqa_data_key in vqa_data.keys():
        if vqa_data_key in model_results:
            print(f"Skipping {vqa_data_key}")
            continue
        
        unique_vqa_data = vqa_data[vqa_data_key].copy()
        correct_answer_index, category = unique_vqa_data.pop("Answer index"), unique_vqa_data.pop("Category")

        uid = vqa_data_key.split('_')[0]
        
        # Load and process video frames using command line argument path
        all_frames = []
        for view_id in view_ids:
            video_path = os.path.join(args.video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4")
            frames = load_video(video_path, max_frames_num=6)
            all_frames.extend(frames)
        
        # Get model response
        answer = submit_vqa_to_model(model, tokenizer, image_processor, str(unique_vqa_data), all_frames)
        print(f"model ouput: {answer}")
        correctness, extracted_index = handle_vqa_result(answer, correct_answer_index)
        print(f"extracted_index: {extracted_index}")
        model_results[vqa_data_key] = {"answer": extracted_index, "correctness": correctness}

        # Periodically save results
        with open(save_file_path, 'w') as f:
            json.dump(model_results, f, indent=4)

if __name__ == "__main__":
    main()