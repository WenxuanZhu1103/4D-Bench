import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import sys
import json
from PIL import Image
import argparse
import re
from myutils import utils

def handle_llm_caption_no_index(content):
    tmp = content.lower()
    caption_start = tmp.find("caption:")
    if caption_start != -1:
        return content[caption_start + len("caption:"):].strip()
    return content

def generate_baseline_caption_from_qwen2_vl(model, processor, frames, uid):
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = f"/tmp/qwen2_vl_7b_frame_{uid}_{i}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    
    # Qwen-2-VL specific image processing
    # Create the message with video frames and the VQA question
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
            
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [f"file://{frame_path}" for frame_path in frame_paths],
                },
                {"type": "text", "text": f"{input_text}"},
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
    os.system(f"rm -rf /tmp/qwen2_vl_7b_frame_{uid}_*")
    return output_text[0].strip()


def generate_baseline_caption(video_data_path, results_save_path, model_name, model, processor):
    uids = [d for d in os.listdir(video_data_path) if os.path.isdir(os.path.join(video_data_path, d))]
    
    for uid in uids:
        json_file_path = os.path.join(results_save_path, f"{uid}.json")
        
        # Skip if caption already exists
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                if f"{model_name}_caption_4d_baseline" in json_data:
                    print(f"skip {uid}")
                    continue

        try:
            view_ids = [1, 8, 16]
            frames = []
            for view_id in view_ids:
                frames.extend(utils.get_video_uniform_frames(os.path.join(video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4"), frame_num=6))
                
            response = generate_baseline_caption_from_qwen2_vl(
                model, processor, frames, uid
            )
            
            caption = handle_llm_caption_no_index(response)
            
            # Save caption to JSON
            data = {}
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
            
            data[f"{model_name}_caption_4d_baseline"] = caption
            
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=4)
                
        except Exception as e:
            print(f"Error processing {uid}: {str(e)}")
            with open("./qwen2_vl_failed.txt", "a") as f:
                f.write(f"{uid}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate captions using Qwen2-VL model')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory for caching model weights')
    parser.add_argument('--video_data_path', type=str, required=True, help='Path to video data')
    parser.add_argument('--results_save_path', type=str, required=True, help='Path to save results')
    args = parser.parse_args()

    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    # Load the Qwen model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        device_map="auto", 
        cache_dir=args.cache_dir
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    os.makedirs(args.results_save_path, exist_ok=True)
    
    
    # Generate captions
    generate_baseline_caption(
        video_data_path=args.video_data_path,
        results_save_path=args.results_save_path,
        model_name=model_name.strip().split('/')[-1].strip(),
        model=model,
        processor=processor
    )
