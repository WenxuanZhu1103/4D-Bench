import os
import json
import torch
import time
import random
import numpy as np
import argparse
import pandas as pd
import base64
from PIL import Image
import torch.backends.cudnn as cudnn
from minigpt4.common.eval_utils import init_model
from minigpt4.conversation.conversation import CONV_VISION
import yaml
from myutils import utils
from torchvision import transforms
import re
import sys
import re

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def match_frames_and_subtitles(frames):
    transform=transforms.Compose([
            transforms.ToPILImage(),
        ])
    
    img_placeholder = ""
    imgs = []
    for frame in frames:
        frame = transform(frame[:,:,::-1]) 
        frame = vis_processor(frame)
        imgs.append(frame)
        img_placeholder += '<Img><ImageHere>'
    
    return torch.stack(imgs), img_placeholder


def prepare_input(frames, input_text):
    frames_features, input_placeholder = match_frames_and_subtitles(frames)
    input_placeholder+="\n"+input_text
    return frames_features, input_placeholder



def submit_vqa_to_minigpt4(model, prepared_images, prepared_instruction):
    length = len(prepared_images)
    prepared_images=prepared_images.unsqueeze(0)
    conv = CONV_VISION.copy()
    conv.system = ""
    conv.append_message(conv.roles[0], prepared_instruction)
    conv.append_message(conv.roles[1], None)
    prompt = [conv.get_prompt()]

    setup_seeds(50)
    answers = model.generate(prepared_images, prompt, max_new_tokens=512, do_sample=True, lengths=[length], num_beams=1)
    print("Generated_answer :", answers[0])
    return answers[0]



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
    args = get_arguments()
    
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    video_data_path = args.video_data_path

    # Initialize MiniGPT-4 model and processors
    global model, vis_processor
    model, vis_processor, *_ = init_model(args)

    vqa_file_path = args.vqa_file_path
    save_file_path = os.path.join(save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")

    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}

    with open(vqa_file_path, 'r') as f:
        vqa_data = json.load(f)

    view_ids = [1, 8, 16]

    for key, unique_vqa_data in vqa_data.items():
        if key in model_results:
            print(f"Skipping {key}...")
            continue

        correct_answer_index, category = unique_vqa_data.pop("Answer index"), unique_vqa_data.pop("Category")

        uid = key.split('_')[0]

        
        frames = []
        for view_id in view_ids:
            frames.extend(utils.get_video_uniform_frames(os.path.join(video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4"), frame_num=6))


        input_text =  f"You are a excellent video analyst. I provide you 18 frames with every six images uniformly sampled from one video, each video captured from a different angle and a question and four choices. \
                        Carefully watch the provided videos and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question. Here is the question and choices: \n {unique_vqa_data}. \n \
                        You must return only the option identifier (e.g., '(A)') without any additional text, do not add any additional analysis, just return the correct option identifier."                       
                        
                        
    
        prepared_images, prepared_instruction = prepare_input(frames, input_text)

        answer = submit_vqa_to_minigpt4(model, prepared_images, prepared_instruction)
        correctness, extracted_index = handle_vqa_result(answer, correct_answer_index)
        print(f"extracted_index: {extracted_index}")
        model_results[key] = {"answer": extracted_index, "correctness": correctness}
        with open(save_file_path, 'w') as f:
            json.dump(model_results, f, indent=4)

    print("VQA Evaluation Completed.")



def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--save_path", required=True, help="Path to save results")
    parser.add_argument("--vqa_file_path", required=True, help="Path to VQA data file")
    parser.add_argument("--video_data_path", required=True, help="Path to video data directory")
    parser.add_argument("--cfg-path", default="./minigpt4/text_configs/mistral_test_config.yaml", help="Path to config file")
    parser.add_argument("--ckpt", default="/ibex/project/c2191/wenxuan_proj/huggingface_model_weight/video_mistral_checkpoint_last.pth", help="Path to model checkpoint")
    parser.add_argument("--add_subtitles", action='store_true', help="whether to add subtitles")
    parser.add_argument("--stream", action='store_true', help="whether to stream the answer")
    parser.add_argument("--question", type=str, help="question to ask")
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--options", nargs="+", help="override some settings...")
    return parser.parse_args()

if __name__ == "__main__":
    main()
