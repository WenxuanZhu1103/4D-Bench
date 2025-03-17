from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import argparse
import os
import json
import numpy as np
import warnings
from decord import VideoReader, cpu
import torch
import copy
import sys
import re

warnings.filterwarnings("ignore")

class LLaVAVideoEvaluator:
    def __init__(self, pretrained_model="lmms-lab/LLaVA-Video-72B-Qwen2", device="cuda", cache_dir=None):
        self.device = device
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            pretrained_model, 
            None, 
            "llava_qwen", 
            torch_dtype="bfloat16", 
            device_map="auto",
            cache_dir=cache_dir
        )
        self.model.eval()

    def load_video(self, video_path, max_frames_num=6):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        
        # Uniformly sample exactly 6 frames
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
        frames = vr.get_batch(frame_idx).asnumpy()
        
        return frames, frame_time_str, video_time

    def process_frames(self, frames):
        processed_frames = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.device).bfloat16()
        return processed_frames

    def generate_response(self, processed_frames, vqa, video_time, frame_info):
        vqa = f"You are a excellent video analyst. I provide you 18 frames with every six images uniformly sampled from one video, each video captured from a different angle and a question and four choices. \
                    Carefully watch the provided videos and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question. Here is the question and choices: \n {vqa}.\n \
                    You must return only the option identifier (e.g., '(A)') without any additional text, do not add any additional analysis, just return the correct option identifier."                       
                    
        # time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(processed_frames)} frames are uniformly sampled from it. These frames are located at {frame_info}."
        full_question = f"{DEFAULT_IMAGE_TOKEN}\n{vqa}"
        
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], full_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        
        outputs = self.model.generate(
            input_ids,
            images=[processed_frames],
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(response)
        return response
    
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
    # Add argument parser
    parser = argparse.ArgumentParser(description='LLaVA Video Evaluation')
    parser.add_argument('--save_path', required=True, help='Path to save results')
    parser.add_argument('--vqa_file_path', required=True, help='Path to VQA data file')
    parser.add_argument('--video_data_path', required=True, help='Path to video data directory')
    parser.add_argument('--cache_dir', required=True, help='Path to cache directory for model weights')
    args = parser.parse_args()

    # Update paths using command line arguments
    os.makedirs(args.save_path, exist_ok=True)
    save_file_path = os.path.join(args.save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")
    
    # Initialize evaluator with cache_dir
    evaluator = LLaVAVideoEvaluator(cache_dir=args.cache_dir)
    
    # Load or create results dictionary
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}
    
    # Load VQA data using argument
    with open(args.vqa_file_path, 'r') as f:
        vqa_data = json.load(f)
    
    view_ids = [1, 8, 16]
    max_frames_per_view = 6

    for vqa_data_key in vqa_data.keys():
        print(vqa_data_key)
        
        if vqa_data_key in model_results:
            print(f"Skipping {vqa_data_key}")
            continue
        
        # Process VQA data
        unique_vqa_data = vqa_data[vqa_data_key].copy()
        correct_answer_index, category = unique_vqa_data.pop("Answer index"), unique_vqa_data.pop("Category")

        uid = vqa_data_key.split('_')[0]
        
        # Process all video views using video_data_path argument
        all_frames = []
        all_frame_times = []
        total_video_time = 0
        
        for view_id in view_ids:
            video_path = os.path.join(args.video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4")
            frames, frame_time, video_time = evaluator.load_video(
                video_path, 
                max_frames_per_view
            )
            all_frames.extend(frames)
            all_frame_times = frame_time
            total_video_time = video_time
        
        # Process all frames together
        processed_frames = evaluator.process_frames(np.array(all_frames))
        
        # Generate response
        frame_info = " | ".join([f"View {view_id}: {times}" for view_id, times in zip(view_ids, all_frame_times)])
        answer = evaluator.generate_response(
            processed_frames,
            str(unique_vqa_data),
            total_video_time,
            frame_info
        )
        
        print(f"model ouput: {answer}")
        correctness, extracted_index = handle_vqa_result(answer, correct_answer_index)
        print(f"extracted_index: {extracted_index}")
        model_results[vqa_data_key] = {"answer": extracted_index, "correctness": correctness}
        
            
        
        with open(save_file_path, 'w') as f:
            json.dump(model_results, f, indent=4)

if __name__ == "__main__":
    main()