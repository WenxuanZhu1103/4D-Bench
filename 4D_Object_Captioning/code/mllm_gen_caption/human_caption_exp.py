import os
import json
import sys
import argparse
import random
import pandas as pd




def generate_baseline_caption(results_save_path, model_name, video_data_path, human_annotations_file):
    """Processes each UID to generate captions using OpenAI."""
    human_annotations = pd.read_csv(human_annotations_file)
    uids = os.listdir(video_data_path)
    for uid in uids:
        json_file_path = os.path.join(results_save_path, f"{uid}.json")
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            if f"{model_name}_caption_4d_baseline" in json_data:
                print(f"skip {uid}")
                continue

        caption_idx = random.randint(1, 5)
        selected_row = human_annotations[human_annotations['folder_name'] == uid]
        if selected_row.empty:
            print(f"No annotation found for {uid}")
            continue
        
        human_caption = selected_row[f'caption_{caption_idx}'].values[0]
        caption = {"human_caption": human_caption, "caption_idx": caption_idx}

        data = {}
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        
        data[f"{model_name}_caption_4d_baseline"] = caption
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, required=False, help='Cache directory path')
    parser.add_argument('--api_key', type=str, required=False, help='OpenAI API key')
    parser.add_argument('--video_data_path', type=str, required=True, help='Path to video data')
    parser.add_argument('--results_save_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--human_annotations', type=str, required=True, help='Path to human annotations')

    args = parser.parse_args()

    model_name = 'human'


    if not os.path.exists(args.video_data_path):
        print(f"Data not exist")
        sys.exit(1)

    os.makedirs(args.results_save_path, exist_ok=True)


    print("Generating baseline captions")
    generate_baseline_caption(args.results_save_path, model_name, args.video_data_path, args.human_annotations)
