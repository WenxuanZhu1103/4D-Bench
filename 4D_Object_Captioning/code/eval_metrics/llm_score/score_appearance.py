from openai import OpenAI 
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
import pandas as pd
import random
import argparse


def annotate(client, human_caption, predicted_caption, model_name):
    response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": 
                "You are an expert in evaluating the quality of video captions. \
                Your task is to rate the predicted caption in terms of recall and precision of visual elements(appearance and shape) in the video with reference to the human-annotated caption. \
                Focus only on whether the predicted caption accurately and completely contains the information from the human-annotated caption. \
                Note you just need to focus on the visual elements. \
                Consider synonyms or paraphrases as valid matches.\
                Provide your evaluation as a matching score where the score is an integer value between 0 and 5. Here is the rating scale: \n \
                Score 5: The predicted caption accurately identifies the object in the video, including its appearance and shape. The caption provides a precise and complete description of the object without missing any significant visual details. \n \
                Score 4: The predicted caption mostly identifies the object accurately, with minor omissions or differences in the description of the appearance or shape. Paraphrases are acceptable, and the overall description is correct, though it may lack some finer details. \n \
                Score 3: The caption identifies some key aspects of the object but misses or incorrectly describes certain visual elements, such as the appearance or shape. There are noticeable gaps, but the overall object is still somewhat recognizable in the caption. \n \
                Score 2: The predicted caption contains several inaccuracies in describing the object’s appearance or shape. While some parts may be correct, the overall description is incomplete or misleading. Precision and recall of visual elements are low. \n \
                Score 1: The caption provides an incorrect description of the object, with major inaccuracies in identifying the appearance and shape. The object is either misidentified or described in a way that does not match the video. \n \
                Score 0: The caption is entirely incorrect, failing to identify the object or its appearance and shape. No valid matches to the human-annotated caption are present.  \
                Here are some rating examples: \n \
                Example 1:{ Human_Caption: 'A red wrecking ball with black chains swings into a big brown cube sitting on a metallic surface that scatters into smaller cubes after being hit'. Predicted Caption: 'a cube and ball connected by a chain'. Score: {'appearance_score': 1} } \n \
                Example 2:{ Human_Caption: 'A woman wearing a pair of combat pants and a tank top throwing a punch'. Predicted Caption: 'a woman in a boxing outfit, wearing a hat, hoodie, and camouflage pants, holding a gun'. Score: {'appearance_score': 3} } \n \
                Example 3:{ Human_Caption: 'Azerbaijan flag that moves with the wind'. Predicted Caption: 'the Azerbaijan flag waving in the wind and a colorful kite'. Score: {'appearance_score': 2} } \n \
                Example 4:{ Human_Caption: '3D model of arms with gray sleeves carrying a gray pistol with brown grip and gray barrel that loads it, fires two bullets, then unloads it'. Predicted Caption: 'A pair of human-like arms in a dark grey sweater holding a handgun with a brown grip and black barrel'. Score: {'appearance_score': 4} } \n \
                Example 5:{ Human_Caption: '3D model of a boy wearing glasses dancing dressed in a grey hood, black pants, gray shoes, he puts on a red cap and a blue backpack'. Predicted Caption: 'a person wearing a pink hat, holding a sword, and surrounded by a glider, bird, and windmill, all adorned with pink hats'. Score: {'appearance_score': 0} } \n \
                Example 6:{ Human_Caption: 'A 3D model of a lightsaber which is emitting blue saber'. Predicted Caption: 'light saber, and flashlight'. Score: {'appearance_score': 5} } "                  
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based captions:\n\n"
                    f"Human-annotated Caption: {human_caption}\n"
                    f"Predicted Caption: {predicted_caption}\n\n"
                    "Please generate the response in the form of a dictionary string with keys 'appearance_score', where its value is the factual accuracy score in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. You must follow this command!"
                    "For example, your response should look like this: {'appearance_score': 4}."
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


            key_name = method + f'_{model_name}_appearance_score'
            
            if key_name in write_json_data.keys():
                print(f"skip {uid}")
                continue

            if not isinstance(pred_caption, str):  # 检查caption是否为字符串
                print(f"Warning: Invalid caption format for {method}")
                continue
                
            try:
                response_dict = annotate(client, human_caption=gt_caption, predicted_caption=pred_caption, model_name=model_name)
                write_json_data[key_name] = response_dict['appearance_score']
            except Exception as e:
                print(f"Error processing caption {method}: {str(e)}")
                continue


            
        with open(write_file_path, 'w') as f:
            json.dump(write_json_data, f, indent=4)
    
if __name__ == "__main__":
    main()


