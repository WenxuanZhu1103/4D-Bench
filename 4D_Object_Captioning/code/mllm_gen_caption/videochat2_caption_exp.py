from video_chat2.utils.config import Config

import io
import os
from video_chat2.models import VideoChat2_it_mistral
from video_chat2.utils.easydict import EasyDict
import torch
import cv2

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
import argparse
decord.bridge.set_bridge("torch")

from myutils import utils



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
        
def handle_llm_caption_no_index(content):
    tmp = content.lower()
    caption_start = tmp.find("caption:")
    if caption_start != -1:
        return content[caption_start + len("caption:"):].strip()
    return content


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
        torch.tensor([29871, 2]).to("cuda:0")]  # '</s>' can be encoded in two different ways.
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
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
#     output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate captions using VideoChat2 model')
    parser.add_argument('--video_data_path', type=str, required=True, help='Path to video data')
    parser.add_argument('--results_save_path', type=str, required=True, help='Path to save results')
    args = parser.parse_args()

    model_name = "videochat2"
    
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
    print(msg)

    model = model.to(torch.device(cfg.device))
    model = model.eval()
    
    uids = [d for d in os.listdir(args.video_data_path) if os.path.isdir(os.path.join(args.video_data_path, d))]
    for uid in uids:
        json_file_path = os.path.join(args.results_save_path, f"{uid}.json")
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
            "roles": ("[INST]", "[/INST]"),
            "messages": [],
            "sep": ""
        })

        chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
        ask(f"{input_text}", chat)

        caption = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=512, print_res=False)[0]
        print(caption)
        
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
