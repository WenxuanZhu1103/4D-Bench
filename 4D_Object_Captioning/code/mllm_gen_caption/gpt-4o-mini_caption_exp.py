import os
import cv2
from openai import OpenAI 
import numpy as np
import json
import time
import pandas as pd
from IPython.display import Image
import base64
import sys
import argparse
from myutils import utils

def handle_llm_caption_no_index(content):
    tmp = content.lower()
    caption_start = tmp.find("caption:")
    if caption_start != -1:
        return content[caption_start + len("caption:"):].strip()
    return content

def generate_baseline_caption_from_openai(client, model_name, input_text, frames):
    base64Frames = []
    for frame in frames:
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frame = base64.b64encode(buffer).decode("utf-8")
        base64Frames.append(base64_frame)
            
    response = client.chat.completions.create(
        model=model_name,
        messages=[
        {"role": "system", "content": "You are an AI assistant that generates descriptive captions for multi-view videos."},
        {"role": "user", "content": [
            f"{input_text}",
            *map(lambda x: {"type": "image_url", 
                            "image_url": {"url": f'data:image/jpg;base64,{x}'}}, base64Frames)
            ]
        }
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content    
    

def generate_baseline_caption(client, results_save_path, video_data_path, model_name):
    """Processes each UID to generate captions using OpenAI."""
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

        frames = []
        view_ids = [1, 8, 16]
        for view_id in view_ids:
            frames.extend(utils.get_video_uniform_frames(os.path.join(video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4"), frame_num=6))
            
        caption = generate_baseline_caption_from_openai(
            client, model_name, input_text, frames
        )
        caption = handle_llm_caption_no_index(caption)

        if caption is None:
            print(f"Failed to generate caption for {uid}")
            with open("openai_failed.txt", "a") as f:
                f.write(f"{uid}\n")
            continue

        data = {}
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        
        data[f"{model_name}_caption_4d_baseline"] = caption
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, required=False, help='Cache directory path')
    parser.add_argument('--api_key', type=str, required=False, help='OpenAI API key')
    parser.add_argument('--video_data_path', type=str, required=True, help='Path to video data')
    parser.add_argument('--results_save_path', type=str, required=True, help='Path to save results')
    args = parser.parse_args()

    model_name = 'gpt-4o-mini'
    api_key = args.api_key
    client = OpenAI(api_key=api_key)

    if not os.path.exists(args.video_data_path):
        print(f"Data not exist")
        sys.exit(1)

    os.makedirs(args.results_save_path, exist_ok=True)


    print("Generating baseline captions")
    generate_baseline_caption(client, args.results_save_path, args.video_data_path, model_name)
