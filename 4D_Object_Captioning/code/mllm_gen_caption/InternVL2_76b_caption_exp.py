import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import math
import argparse
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_video_frames(video_paths, frame_ids, input_size=448):
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    for video_path in video_paths:
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_indices = np.linspace(0, len(vr) - 1, len(frame_ids), dtype=int)
        for idx in frame_indices:
            img = Image.fromarray(vr[idx].asnumpy()).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=1)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
            
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def generate_caption_from_internvl2(model, tokenizer, video_data_path, uid, view_ids, frame_num=6):
    video_paths = [os.path.join(video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4") for view_id in view_ids]
    pixel_values, num_patches_list = load_video_frames(video_paths, range(frame_num))
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(pixel_values))])
    prompt = video_prefix + "I have multiple videos of the object captured from different angles. \
    I provide you 18 images, with every six images uniformly sampled from one video, each video captured from a different angle. \
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

    generation_config = dict(max_new_tokens=128, do_sample=True)
    response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
    print(response.strip())
    return response.strip()

def handle_caption(content):
    tmp = content.lower()
    caption_start = tmp.find("caption:")
    if caption_start != -1:
        return content[caption_start + len("caption:"):].strip()
    return content

def generate_baseline_caption(video_data_path, results_save_path, model_name, model, tokenizer):
    uids = [d for d in os.listdir(video_data_path) if os.path.isdir(os.path.join(video_data_path, d))]
    
    for uid in uids:
        json_file_path = os.path.join(results_save_path, f"{uid}.json")
        
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                if f"{model_name}_caption_4d_baseline" in json_data:
                    print(f"Skip {uid}")
                    continue

        try:
            response = generate_caption_from_internvl2(
                model, tokenizer,
                video_data_path,
                uid,
                view_ids=[1, 8, 16],
                frame_num=6
            )
            
            caption = handle_caption(response)
            
            data = {}
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
            
            data[f"{model_name}_caption_4d_baseline"] = caption
            
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=4)
                    
        except Exception as e:
            print(f"Error processing {uid}: {str(e)}")
            with open("./internvl2_failed.txt", "a") as f:
                f.write(f"{uid}\n")

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate captions using InternVL2 model')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory for caching model weights')
    parser.add_argument('--api_key', type=str, default=None, help='API key if needed')
    parser.add_argument('--video_data_path', type=str, required=True, help='Path to video data')
    parser.add_argument('--results_save_path', type=str, required=True, help='Path to save results')
    args = parser.parse_args()

    model_name = "OpenGVLab/InternVL2-Llama3-76B"
    device_map = split_model('InternVL2-Llama3-76B')
    
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
        cache_dir=args.cache_dir
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    
    os.makedirs(args.results_save_path, exist_ok=True)
    
    generate_baseline_caption(
        video_data_path=args.video_data_path,
        results_save_path=args.results_save_path,
        model_name=model_name.split('/')[-1],
        model=model,
        tokenizer=tokenizer
    )