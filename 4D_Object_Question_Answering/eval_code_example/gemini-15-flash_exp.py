import os
import json
import cv2
import google.generativeai as genai
from PIL import Image
from myutils import utils
import sys
import numpy as np
import io
import re
import argparse

def submit_vqa_to_model(model, vqa, video_data_path, uid, view_ids, frame_num=6):
    # Configure safety settings
    safety_config = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    
    system_prompt = f"You are a excellent video analyst. I provide you 18 frames with every six images uniformly sampled from one video, each video captured from a different angle and a question and four choices. \
            Carefully watch the provided videos and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question. Here is the question and choices: \n {vqa}.\n \
            You must return only the option identifier (e.g., '(A)') without any additional text, do not add any additional analysis, just return the correct option identifier."                       
    
    input_content = [system_prompt]
    imgs = []
    
    for view_id in view_ids:
        video_path = os.path.join(video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4")
        frames = utils.get_video_uniform_frames(video_path, frame_num=frame_num)
        
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            imgs.append(pil_image)
    
    input_content.extend(imgs)
    
    response = model.generate_content(
        input_content,
        safety_settings=safety_config
    )
    try:
        response_text = response.text.strip()
        print(response_text)
        return response_text
    except Exception as e:
        print(e)
    
        return None
        

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
    parser = argparse.ArgumentParser(description='Evaluate VQA model')
    parser.add_argument('--save_path', required=True, help='Path to save results')
    parser.add_argument('--vqa_file_path', required=True, help='Path to VQA data file')
    parser.add_argument('--video_data_path', required=True, help='Path to video data')
    parser.add_argument('--api_key', required=True, help='API key for Gemini model')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    model_name = "gemini-1.5-flash"
    print(f"testing {model_name}...")
    
    # Configure Gemini with command line API key
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel(model_name)
    
    save_file_path = os.path.join(args.save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")
    
    # Load existing results if any
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}
    
    # Load VQA data
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
        print(uid)
        print(unique_vqa_data)
        
        answer = submit_vqa_to_model(model, unique_vqa_data, args.video_data_path, uid, view_ids)
        if answer is None:
            continue
        print(f"model ouput: {answer}")
        correctness, extracted_index = handle_vqa_result(answer, correct_answer_index)
        print(f"extracted_index: {extracted_index}")
        model_results[vqa_data_key] = {"answer": extracted_index, "correctness": correctness}
        
        with open(save_file_path, 'w') as f:
            json.dump(model_results, f, indent=4)