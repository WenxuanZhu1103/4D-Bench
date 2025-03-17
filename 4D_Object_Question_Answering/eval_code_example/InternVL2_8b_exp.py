import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import math
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import re
import argparse
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
    parser = argparse.ArgumentParser(description='InternVL2-8B Video Question Answering')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--vqa_file_path', type=str, required=True, help='Path to VQA data file')
    parser.add_argument('--video_data_path', type=str, required=True, help='Path to video data directory')
    parser.add_argument('--cache_dir', type=str, required=True, help='Path to cache directory for model weights')
    args = parser.parse_args()

    model_name = "OpenGVLab/InternVL2-8B"
    device_map = split_model('InternVL2-8B')
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
    
    os.makedirs(args.save_path, exist_ok=True)
    save_file_path = os.path.join(args.save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")

    # Load or create results dictionary
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}
    
    # Load VQA data
    with open(args.vqa_file_path, 'r') as f:
        vqa_data = json.load(f)
    
    img_view_ids = [1, 8, 16]
    frame_ids = range(6)
    
    for vqa_data_key in vqa_data.keys():
        if vqa_data_key in model_results:
            print(f"Skipping {vqa_data_key}")
            continue
        
        # Process VQA data
        unique_vqa_data = vqa_data[vqa_data_key].copy()
        correct_answer_index, category = unique_vqa_data.pop("Answer index"), unique_vqa_data.pop("Category")

        uid = vqa_data_key.split('_')[0]
        
        video_paths = [os.path.join(args.video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4") for view_id in img_view_ids]
        pixel_values, num_patches_list = load_video_frames(video_paths, frame_ids)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(pixel_values))])
        prompt = video_prefix + f"You are a excellent video analyst. I provide you 18 frames with every six images uniformly sampled from one video, each video captured from a different angle and a question and four choices. \
            Carefully watch the provided videos and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question. Here is the question and choices: \n {unique_vqa_data}.\n \
            You must return only the option identifier (e.g., '(A)') without any additional text, do not add any additional analysis, just return the correct option identifier."                       

        generation_config = dict(max_new_tokens=128, do_sample=True)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        answer = response.strip()
        print(f"model ouput: {answer}")
        correctness, extracted_index = handle_vqa_result(answer, correct_answer_index)
        print(f"extracted_index: {extracted_index}")
        model_results[vqa_data_key] = {"answer": extracted_index, "correctness": correctness}
        with open(save_file_path, 'w') as f:
            json.dump(model_results, f, indent=4)
    
