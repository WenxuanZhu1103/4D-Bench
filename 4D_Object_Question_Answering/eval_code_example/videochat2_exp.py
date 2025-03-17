import argparse
from video_chat2.utils.config import Config
import sys
import io
import os
from video_chat2.models import VideoChat2_it_mistral
from video_chat2.utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from video_chat2.dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

import matplotlib.pyplot as plt

from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy

import json
from collections import OrderedDict

from tqdm import tqdm

import time
import decord
decord.bridge.set_bridge("torch")
from myutils import utils
import cv2
import re



def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
#         seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text])
        


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to("cuda:0"),
        torch.tensor([29871, 2]).to("cuda:0")]  # 'ursor' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token 
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token  at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('ursor')[0]  # remove the stop sign 
#     output_text = output_text.split('ursor')[-1].strip()
    conv.messages[-1][1] = output_text + 'ursor'
    return output_text, output_token.cpu().numpy()


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(frames, resolution=512):
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = []
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        images_group.append(pil_img)
        
    torch_imgs = transform(images_group)
    return torch_imgs

    

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    # print(f"n_position: {n_position}")
    # print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            # print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            # print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        # print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        # print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', required=True, help='Path to save results')
    parser.add_argument('--vqa_file_path', required=True, help='Path to VQA data file')
    parser.add_argument('--video_data_path', required=True, help='Path to video data')
    parser.add_argument('--api_key', type=str, default=None, help='API key if needed')
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    model_name = "videochat2"
    save_file_path = os.path.join(args.save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")
    
    config_file = "./video_chat2/configs/config_mistral.json"
    cfg = Config.from_file(config_file)
    cfg.model.vision_encoder.num_frames = 4
    model = VideoChat2_it_mistral(config=cfg.model)

    # add lora to run stage3 model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head"
        ]
    )
    model.mistral_model = get_peft_model(model.mistral_model, peft_config)
    state_dict = torch.load(cfg.model.videochat2_model_path_stage3, "cpu")
    if 'model' in state_dict.keys():
        msg = model.load_state_dict(state_dict['model'], strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)

    model = model.to(torch.device(cfg.device))
    model = model.eval()
    
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}
    
    view_ids = [1, 8, 16]
    with open(args.vqa_file_path, 'r') as f:
        vqa_data = json.load(f)
    vqa_data_keys = vqa_data.keys()
    
    for vqa_data_key in vqa_data_keys:
        unique_vqa_data = vqa_data[vqa_data_key]
        correct_answer_index, category = unique_vqa_data.pop("Answer index"), unique_vqa_data.pop("Category")

        uid = vqa_data_key.split('_')[0]
        frames = []
        for view_id in view_ids:
            frames.extend(utils.get_video_uniform_frames(os.path.join(args.video_data_path, uid, f"view_{view_id}_rgb_white_bg.mp4"), frame_num=6))
    
        vid = load_video(frames)
        num_frame = 18
        resolution = 512
        new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
        model.vision_encoder.encoder.pos_embed = new_pos_emb

            
        TC, H, W = vid.shape
        video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")

        img_list = []
        with torch.no_grad():
            image_emb, _ = model.encode_img(video, "Watch the video and answer the question.")

        img_list.append(image_emb)

        chat = EasyDict({
            "system": "",
            "roles": ("ursor", "ursor"),
            "messages": [],
            "sep": ""
        })
        input_text = f"You are a excellent video analyst. I provide you 18 frames with every six images uniformly sampled from one video, each video captured from a different angle and a question and four choices. \
            Carefully watch the provided videos and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question. Here is the question and choices: \n {unique_vqa_data}.\n \
            You must return only the option identifier (e.g., '(A)') without any additional text, do not add any additional analysis, just return the correct option identifier."                       
            
        chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> ursor"])
        ask(f"{input_text}", chat)
        response = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=512, print_res=False)[0]
        print(f"model ouput: {response}")
        correctness, extracted_index = handle_vqa_result(response, correct_answer_index)
        print(f"extracted_index: {extracted_index}")
        
        # Save results directly to JSON file without creating subdirectories
        model_results[vqa_data_key] = {"answer": extracted_index, "correctness": correctness}
        with open(save_file_path, 'w') as f:
            json.dump(model_results, f, indent=4)
    


    

        

    
