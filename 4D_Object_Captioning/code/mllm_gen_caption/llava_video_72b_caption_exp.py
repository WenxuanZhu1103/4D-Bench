from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import os
import sys
import json
import time
import torch
from PIL import Image
import numpy as np
from decord import VideoReader, cpu
from operator import attrgetter
import copy
import argparse

def handle_llm_caption_no_index(content):
    tmp = content.lower()
    caption_start = tmp.find("caption:")
    if caption_start != -1:
        return content[caption_start + len("caption:"):].strip()
    return content


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


def generate_baseline_caption_from_llava(model, tokenizer, image_processor, frames, device="cuda"):
    image_tensors = []
        
    processed_frames = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(device)
    image_tensors.append(processed_frames)

    conv_template = "qwen_1_5"
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

    question = f"{DEFAULT_IMAGE_TOKEN}\n{input_text}"

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in frames]

    output_ids = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(response)
    return response

def generate_baseline_caption(results_save_path, model_name, model, tokenizer, image_processor):
    uids = [d for d in os.listdir(results_save_path) if os.path.isdir(os.path.join(results_save_path, d))]
    for uid in uids:
        json_file_path = os.path.join(results_save_path, f"{uid}.json")
        
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                if f"{model_name}_caption_4d_baseline" in json_data:
                    print(f"skip {uid}")
                    continue

        try:
            all_frames = []
            view_ids = [1, 8, 16]
            for view_id in view_ids:
                video_path = os.path.join(args.video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4")
                frames = load_video(video_path, max_frames_num=6)
                all_frames.extend(frames)
                
            response = generate_baseline_caption_from_llava(
                model, tokenizer, image_processor,
                all_frames
            )
            
            caption = handle_llm_caption_no_index(response)
            print(caption)
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
            with open("./llava_failed.txt", "a") as f:
                f.write(f"{uid}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, required=False, help='Cache directory for model weights')
    parser.add_argument('--api_key', type=str, required=False, help='API key')
    parser.add_argument('--video_data_path', type=str, required=True, help='Path to video data')
    parser.add_argument('--results_save_path', type=str, required=True, help='Path to save results')
    args = parser.parse_args()

    pretrained = "lmms-lab/LLaVA-Video-72B-Qwen2"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", 
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    model.eval()
    
    if not os.path.exists(args.video_data_path):
        print("Video data path does not exist")
        sys.exit(1)
    
    os.makedirs(args.results_save_path, exist_ok=True)
    
    # Generate captions
    generate_baseline_caption(
        args.results_save_path,
        model_name=pretrained.strip().split('/')[-1].strip(),
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor
    )