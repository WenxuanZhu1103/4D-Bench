import os
import cv2
from PIL import Image
import sys
import google.generativeai as genai
import numpy as np
import json
import time
import pandas as pd
import argparse
from myutils import utils


def handle_llm_caption_no_index(content):
    tmp = content.lower()
    caption_start = tmp.find("caption:")
    if caption_start != -1:
        return content[caption_start + len("caption:"):].strip()
    return content


def generate_baseline_caption_from_gemini(gemini_model, input_text, video_data_path, uid, view_ids, frame_num=6):
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
    
    input_content = []
    imgs = []
    input_content.extend([input_text])
    
    for view_id in view_ids:
        video_path = os.path.join(video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4")
        frames = utils.get_video_uniform_frames(video_path, frame_num=frame_num)
        
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            imgs.append(pil_image)

    input_content.extend(imgs)
    response = gemini_model.generate_content(input_content, safety_settings = safety_config)
    return response       


def generate_baseline_caption(video_data_path, results_save_path, model_name):
    uids = os.listdir(video_data_path)
    for uid in uids:
        json_file_path = os.path.join(results_save_path, f"{uid}.json")
        
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                if f"{model_name}_caption_4d_baseline" in json_data:
                    print(f"skip {uid}")
                    continue
            
        input_text = f"I have multiple videos of the object captured from different angles. I provide you 18 images, with every six images uniformly sampled from one video, each video captured from a different angle. \
                Your job is to generate one fluent caption for this multi-view video in English, \
                provide a detailed description of the object's or character's appearance, including shape, color, texture, and any notable features. \
                Additionally, describe the actions taking place, focusing on how the object or character moves and behaves throughout the scene. \
                The caption should not describe the background. \
                You must strictly return in the following format: caption: caption content. \
                Here are some examples: \n \
                Example 1: caption: A young woman with black hair wearing silver jumpsuit is lying on the floor and then gently rises. \n \
                Example 2: caption: A military infantryman in green and brown camouflage gear holds a black pistol in his left hand and dances with his arms and legs moving first to the left then to the right. \n \
                Example 3: caption: A 3D model of a fish pond with blue walls, and brown ground, a fish swims next to a creature that looks like an animal that is lying down. \n \
                Example 4: caption: 3D model of a yellow emoji with closed eyes that sticks out its red tongue and moves from right to left. \n \
                Example 5: caption: A man with brown hair, a moustache and sunglasses wears a green coat, black pants, a white shirt and a black tie walks straight then turns raising his right hand up. "
                
        i = 0
        while(True):
            try:
                i = i + 1
                if i > 1:
                    print(f"{uid} failed")
                    break
                
                response_ = generate_baseline_caption_from_gemini(gemini_model, input_text, video_data_path, uid, view_ids=[1, 8, 16], frame_num=6) 
                if len(response_.candidates) == 0:
                    print(f"{uid} generated content is None")
                    failed_uids_file_path = "./gemini_failed.txt"
                    with open(f"{failed_uids_file_path}", "a", encoding="utf-8") as file:
                        file.write(uid+"\n")
                    break
                
                caption = handle_llm_caption_no_index(response_.text)
                print(caption)
                if caption is None: 
                    continue
                else:
                    data = {}
                    if os.path.exists(json_file_path):
                        with open(json_file_path, 'r') as f:
                            data = json.load(f)
                    
                    data[f"{model_name}_caption_4d_baseline"] = caption
                    
                    with open(json_file_path, 'w') as f:
                        json.dump(data, f, indent=4)
                    break

            except Exception as e:
                print(e)
                time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, required=False, help='Cache directory for model weights')
    parser.add_argument('--api_key', type=str, required=False, help='API key for Gemini')
    parser.add_argument('--video_data_path', type=str, required=True, help='Path to video data')
    parser.add_argument('--results_save_path', type=str, required=True, help='Path to save results')
    args = parser.parse_args()

    model_name = 'gemini-1.5-pro'
    genai.configure(api_key=args.api_key)
    gemini_model = genai.GenerativeModel(model_name)
    
    if not os.path.exists(args.video_data_path):
        print(f"Video data path does not exist")
        sys.exit(1)
        
    os.makedirs(args.results_save_path, exist_ok=True)
    

    generate_baseline_caption(args.video_data_path, args.results_save_path, model_name=model_name)