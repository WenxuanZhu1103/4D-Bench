import os
import json
from openai import OpenAI 
from myutils import utils
import base64
import cv2
import sys
import re
import argparse

def submit_vqa_to_model(client, model_name, vqa, frames):
    # print(vqa)
    base64Frames = []
    for frame in frames:
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frame = base64.b64encode(buffer).decode("utf-8")
        base64Frames.append(base64_frame)
        
    response = client.chat.completions.create(
        model=model_name,
        messages=[
                {
                "role": "system",
                "content": 
                        "You are a excellent video analyst. I will provide you 18 frames with every six images uniformly sampled from one video, each video captured from a different angle and a question and four choices. \
                        Carefully watch the provided videos and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question."
                },
                {
                    "role": "user",
                    "content": [f"Here are the multi-view videos: ", *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames), 
                                f"Question and choices: {vqa}", "You must return only the option identifier (e.g., '(A)') without any additional text, do not add any additional analysis, just return the correct option identifier."]
                }
            ]
    )
    
    response_message = response.choices[0].message.content.strip()
    print(response_message)
    return response_message


def handle_vqa_result(pred_answer, correct_answer_index):
    if not pred_answer or not isinstance(pred_answer, str):
        print("Invalid input: prediction answer is empty or not a string")
        return -2, -1

    # Convert to uppercase
    cleaned_answer = pred_answer
    pattern = r'(?:^|[\s\(\.,;:])(A|B|C|D)(?:[\s\)\.,;:]|$)'
    
    matches = re.findall(pattern, cleaned_answer)
    if matches:
        letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        pred_number = letter_to_number[matches[0]]
        
        if pred_number == correct_answer_index:
            print("Correct answer")
            return 1, pred_number
        else:
            print("Wrong answer")
            return 0, pred_number
            
    print("Invalid response format. No valid option (A/B/C/D) found")
    return -1, -1

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='Evaluate VQA model')
    parser.add_argument('--save_path', required=True, help='Path to save results')
    parser.add_argument('--vqa_file_path', required=True, help='Path to VQA data file')
    parser.add_argument('--video_data_path', required=True, help='Path to video data')
    parser.add_argument('--api_key', required=True, help='API key for OpenAI')
    args = parser.parse_args()
    
    # Use arguments instead of hardcoded paths
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    model_name = "gpt-4o"
    print(f"testing {model_name}...")
    video_data_path = args.video_data_path
    client = OpenAI(api_key=args.api_key)
        
    save_file_path = os.path.join(save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")
    
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}
        
    vqa_file_path = args.vqa_file_path
    view_ids = [1, 8, 16]
    with open(vqa_file_path, 'r') as f:
        vqa_data = json.load(f)
    vqa_data_keys = vqa_data.keys()
    
    flag = False
    for vqa_data_key in vqa_data_keys:
        if vqa_data_key in model_results.keys():
            print(f"skip {vqa_data_key}")
            continue
        
        unique_vqa_data = vqa_data[vqa_data_key]
        correct_answer_index, category = unique_vqa_data.pop("Answer index"), unique_vqa_data.pop("Category")

        uid = vqa_data_key.split('_')[0]
        
        frames = []
        for view_id in view_ids:
            frames.extend(utils.get_video_uniform_frames(os.path.join(video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4"), frame_num=6))
        
        
        answer = submit_vqa_to_model(client, model_name, unique_vqa_data, frames=frames)
        print(f"model ouput: {answer}")
        correctness, extracted_index = handle_vqa_result(answer, correct_answer_index)
        print(f"extracted_index: {extracted_index}")
        model_results[vqa_data_key] = {"answer": extracted_index, "correctness": correctness}
        
            
        
        with open(save_file_path, 'w') as f:
            json.dump(model_results, f, indent=4)