from openai import OpenAI 
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
import pandas as pd
import random

def annotate(client, human_caption, predicted_caption, model_name):

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": 
                        "You are an expert in evaluating the quality of video captions. \
                        Your task is to rate the predicted caption in terms of recall and precision of the object's actions in the video with reference to the human-annotated caption.\
                        Note you just need to focus on the action descriptions in the captions. \
                        Consider synonyms or paraphrases as valid matches. \
                        Provide your evaluation as a matching score where the score is an integer value between 0 and 5. Here is the rating scale: \n \
                        Score 5: The predicted caption accurately identifies the actions of the object in the video, including the sequence, timing, and details of the actions. Synonyms or paraphrases are valid matches. The caption provides a precise and complete description of the actions without missing any significant aspects.\n \
                        Score 4: The predicted caption mostly identifies the actions accurately, with minor omissions or differences in the description of the actions. Paraphrases are acceptable, and the overall description is correct, though it may lack some finer details. \n \
                        Score 3: The caption identifies some key actions but misses or incorrectly describes certain details, such as timing, order, or subtle movements. There are noticeable gaps, but the overall actions are still somewhat recognizable in the caption. \n \
                        Score 2: The predicted caption contains several inaccuracies in describing the object's actions. While some parts may be correct, the overall description is incomplete or misleading. Precision and recall of actions are low. \n \
                        Score 1: The caption provides an incorrect description of the object's actions, with major inaccuracies in identifying the actions or their sequence. The actions are either misidentified or described in a way that does not match the video. \n \
                        Score 0: The caption is entirely incorrect, failing to identify the object's actions. No valid matches to the human-annotated caption are present. \n \
                        Here are some rating examples: \n \
                        Example 1:{ Human_Caption: '3D model of a woman covered in white and purple mesh is warming up and shadow boxing'. Predicted Caption:' a figure with a purple and black grid-like texture is running in place, their arms swinging at their sides and their legs lifting up alternately.' Score: {'action_score': 1} } \n \
                        Example 2:{ Human_Caption: 'A white and yellow star wars sitting on his knees squatting,stretches his right arm and back'. Predicted Caption:' this is a 3d model of a clone trooper with yellow markings on his helmet, shoulders, knees, and shins. he is crouching down on one knee, wearing white armor with grey accents and a utility belt. the 327th star corps emblem is visible on his left shoulder.' Score: {'action_score': 3} } \n \
                        Example 3:{ Human_Caption: 'Black puppy with white nose wiggling its tail.' Predicted Caption: 'a low-poly dog with a black body and white paws and face stands still. its tail is black, and its ears are floppy. the dog is rendered in a minimalist style. it remains stationary throughout the scene.' Score: {'action_score': 2} } \n \
                        Example 4:{ Human_Caption: 'A ninja-looking robot in black and red armor with a shield and sword is jumping up, twisting and slashing the air with his sword before landing down.' Predicted Caption: 'a red and black armored warrior, adorned with a demonic mask, engages in a display of martial prowess, wielding both a gleaming sword and a circular shield with a blue emblem. they leap, twirl, and strike dynamic poses, their movements fluid and controlled.' Score: {'action_score': 4} } \n \
                        Example 5:{ Human_Caption: 'A 3D model of a green turtle with a brown shell swimming.' Predicted Caption: 'a low-poly 3d model of a green sea turtle with a brown shell.' Score: {'action_score': 0} } \n \
                        Example 6:{ Human_Caption: 'An animated bearded man with brown hair, red beanie and a flannel shirt is wielding an axe in his right hand while running.' Predicted Caption: 'a cartoon lumberjack with a big black beard, wearing a red beanie hat, red and black plaid shirt, blue pants, brown boots, and brown gloves runs while carrying an axe.' Score: {'action_score': 5} } "                  
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based captions:\n\n"
                    f"Human_Caption: {human_caption}\n"
                    f"Predicted Caption: {predicted_caption}\n\n"
                    "Please generate the response in the form of a dictionary string with keys 'action_score', where its value is the factual accuracy score in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. You must follow this command!"
                    "For example, your response should look like this: {'action_score': 4}."
            }
        ]
    )
    response_message = response.choices[0].message.content
    print(response_message)
    response_dict = ast.literal_eval(response_message)
    return response_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_annotations_path", type=str, required=True, help="Path to human annotations CSV file")
    parser.add_argument("--caption_path", type=str, required=True, help="Path to caption results")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    args = parser.parse_args()

    model_name = "gpt-4o"
    random.seed(42)
    human_annotations = pd.read_csv(args.human_annotations_path)
    client = OpenAI(api_key=args.api_key)
    caption_path = args.caption_path
    save_path = os.path.join(os.path.dirname(caption_path), f"{os.path.basename(caption_path)}_scores")
    os.makedirs(save_path, exist_ok=True)  # 添加创建目录的代码
    
    files = os.listdir(caption_path)
    for file in files:
        uid = file.split('.')[0]
        selected_row = human_annotations[human_annotations['folder_name'] == uid]
        if len(selected_row) == 0:  # 检查是否找到对应的行
            print(f"Warning: No data found for uid {uid}")
            continue
            
        
        read_file_path = os.path.join(caption_path, file)
        if not os.path.exists(read_file_path):
            print(f"Warning: Input file not found: {read_file_path}")
            continue
            
        write_file_path = os.path.join(save_path, file)
        
        try:
            with open(read_file_path, 'r') as f:
                read_json_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON file: {read_file_path}")
            continue
            
        methods = list(read_json_data.keys())
            
        write_json_data = {}
        if os.path.exists(write_file_path):
            with open(write_file_path, 'r') as f:
                write_json_data = json.load(f)
                
        for method in methods:            
            if "human" in method:
                human_idx = read_json_data[method]["caption_idx"]
                pred_caption = read_json_data[method]["human_caption"]
                available_indices = list(range(1, 6))
                available_indices.remove(human_idx)
                gt_idx = random.choice(available_indices)
                gt_caption = selected_row[f'caption_{gt_idx}'].values[0]

            else:
                gt_caption = selected_row[f'caption_{random.randint(1, 5)}'].values[0]
                pred_caption = read_json_data[method]


            key_name = method + f'_{model_name}_action_score'
            
            if key_name in write_json_data.keys():
                print(f"skip {uid}")
                continue

            if not isinstance(pred_caption, str):  # 检查caption是否为字符串
                print(f"Warning: Invalid caption format for {method}")
                continue
                
            try:
                response_dict = annotate(client, human_caption=gt_caption, predicted_caption=pred_caption, model_name=model_name)
                write_json_data[key_name] = response_dict['action_score']
            except Exception as e:
                print(f"Error processing caption {method}: {str(e)}")
                continue
            
        with open(write_file_path, 'w') as f:
            json.dump(write_json_data, f, indent=4)
    
if __name__ == "__main__":
    main()
