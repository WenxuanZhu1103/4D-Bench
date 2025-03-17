import torch
import os
from minigpt4.common.eval_utils import init_model
from minigpt4.conversation.conversation import CONV_VISION
import json
import argparse
from PIL import Image
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn 
import time
import yaml 
import sys
import pandas as pd
from torchvision import transforms
from myutils import utils

def handle_llm_caption_no_index(content):
    tmp = content.lower()
    caption_start = tmp.find("caption:")
    if caption_start != -1:
        return content[caption_start + len("caption:"):].strip()
    return content


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


def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']
    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)
    return text

def model_generate(*args, **kwargs):
    # for 8 bit and 16 bit compatibility
    with model.maybe_autocast():
        output = model.llama_model.generate(*args, **kwargs)
    return output

  
def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--cfg-path", help="path to configuration file.", default="./minigpt4/text_configs/mistral_test_config.yaml")
    parser.add_argument("--ckpt", type=str,default='/ibex/project/c2191/wenxuan_proj/huggingface_model_weight/video_mistral_checkpoint_last.pth', help="path to checkpoint")
    parser.add_argument("--add_subtitles",action= 'store_true',help="whether to add subtitles")
    parser.add_argument("--stream",action= 'store_true',help="whether to stream the answer")
    parser.add_argument("--question", type=str, help="question to ask")
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--video_data_path", type=str, required=True, help="Path to video data")
    parser.add_argument("--results_save_path", type=str, required=True, help="Path to save results")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
                "in xxx=yyy format will be merged into config file (deprecate), "
                "change to --cfg-options instead.",
    )
    return parser.parse_args()

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
def generate_baseline_caption_from_minigpt4_video(input_text, frames):
    prepared_images, prepared_instruction = prepare_input(frames, input_text)
    if prepared_images is None:
        return "Video cann't be open ,check the video path again"
    length = len(prepared_images)
    prepared_images=prepared_images.unsqueeze(0)
    conv = CONV_VISION.copy()
    conv.system = ""
    # if you want to make conversation comment the 2 lines above and make the conv is global variable
    conv.append_message(conv.roles[0], prepared_instruction)
    conv.append_message(conv.roles[1], None)
    prompt = [conv.get_prompt()]

    setup_seeds(50)
    answers = model.generate(prepared_images, prompt, max_new_tokens=args.max_new_tokens, do_sample=True, lengths=[length], num_beams=1)
    print("Generated caption :", answers[0])
    return answers[0]


def generate_baseline_caption(video_data_path, results_save_path, model_name):
    uids = [d for d in os.listdir(video_data_path) if os.path.isdir(os.path.join(video_data_path, d))]
    for uid in uids:
        json_file_path = os.path.join(results_save_path, f"{uid}.json")
            
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
                
                frames = []
                view_ids = [1, 8, 16]
                for view_id in view_ids:
                    frames.extend(utils.get_video_uniform_frames(os.path.join(video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4"), frame_num=6))
                    
                response_ = generate_baseline_caption_from_minigpt4_video(input_text, frames) 
                caption = handle_llm_caption_no_index(response_)
                if caption is None: 
                    continue
                else:
                    data = {}
                    if os.path.exists(json_file_path):
                        with open(json_file_path, 'r') as json_file:
                            data = json.load(json_file)
                            
                    data[f"{model_name}_caption_4d_baseline"] = caption
                    with open(json_file_path, 'w') as json_file:
                        json.dump(data, json_file, indent=4)
                    break

            except Exception as e:
                print(e)
                time.sleep(10)


if __name__ == "__main__":
    args = get_arguments()
    model, vis_processor, whisper_gpu_id, minigpt4_gpu_id, answer_module_gpu_id = init_model(args)
    conv = CONV_VISION.copy()
    conv.system = ""
    
    if not os.path.exists(args.video_data_path):
        print(f"Video data path does not exist")
        sys.exit(1)
        
    os.makedirs(args.results_save_path, exist_ok=True)
    
    
    generate_baseline_caption(args.video_data_path, args.results_save_path, model_name="minigpt4-video")
