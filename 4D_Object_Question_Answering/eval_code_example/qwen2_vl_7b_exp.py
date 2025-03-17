import os
import json
import cv2
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from myutils import utils
import sys
import re

def submit_vqa_to_qwen(model, processor, vqa, frames):
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = f"/tmp/frame_{i}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

    print(vqa)
    vqa = f"You are a excellent video analyst. I provide you 18 frames with every six images uniformly sampled from one video, each video captured from a different angle and a question and four choices. \
            Carefully watch the provided videos and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question. Here is the question and choices: \n {vqa}.\n \
            You must return only the option identifier (e.g., '(A)') without any additional text, do not add any additional analysis, just return the correct option identifier."                       
                    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [f"file://{frame_path}" for frame_path in frame_paths],
                },
                {"type": "text", "text": f"{vqa}"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Perform inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])
    os.system("rm -rf /tmp/frame_*")
    return output_text[0].strip()


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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate VQA using Qwen2-VL model')
    parser.add_argument('--save_path', required=True, help='Path to save results')
    parser.add_argument('--vqa_file_path', required=True, help='Path to VQA data file')
    parser.add_argument('--video_data_path', required=True, help='Path to video data directory')
    parser.add_argument('--cache_dir', required=True, help='Path to cache directory for model weights')
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    # Load the Qwen model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", cache_dir=args.cache_dir
    )
    processor = AutoProcessor.from_pretrained(model_name)

    save_file_path = os.path.join(args.save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}

    view_ids = [1, 8, 16]
    with open(args.vqa_file_path, 'r') as f:
        vqa_data = json.load(f)
    vqa_data_keys = vqa_data.keys()

    for vqa_data_key in vqa_data_keys:
        if vqa_data_key in model_results.keys():
            print(f"skip {vqa_data_key}")
            continue
        
        unique_vqa_data = vqa_data[vqa_data_key]
        correct_answer_index, category = unique_vqa_data.pop("Answer index"), unique_vqa_data.pop("Category")

        uid = vqa_data_key.split('_')[0]
        frames = []
        for view_id in view_ids:
            frames.extend(utils.get_video_uniform_frames(os.path.join(args.video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4"), frame_num=6))

        answer = submit_vqa_to_qwen(model, processor, unique_vqa_data, frames=frames)

        correctness, extracted_index = handle_vqa_result(answer, correct_answer_index)
        print(f"extracted_index: {extracted_index}")
        model_results[vqa_data_key] = {"answer": extracted_index, "correctness": correctness}

        with open(save_file_path, 'w') as f:
            json.dump(model_results, f, indent=4)
