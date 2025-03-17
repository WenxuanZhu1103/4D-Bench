import pandas as pd
import os
import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from pycocoevalcap.bleu.bleu import Bleu
from cidereval import cider, ciderD
import argparse
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from pycocoevalcap.rouge.rouge import Rouge
from bert_score import score as bert_score
import torch

def compute_scores(gt_captions, candidate):
    references = {0: gt_captions}
    candidates = {0: [candidate]}

    # Initialize evaluators
    bleu = Bleu(4)
    rouge = Rouge()

    # Compute scores
    cider_score = np.max(cider([candidate], [gt_captions])['scores'])
    bleu_score, _ = bleu.compute_score(references, candidates)
    meteor_score = nltk_meteor([c.split() for c in gt_captions], candidate.split())
    rouge_score, _ = rouge.compute_score(references, candidates)

    return cider_score, bleu_score, meteor_score, rouge_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_save_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--human_annotations_path", type=str, required=True, help="Path to human annotations CSV file")
    args = parser.parse_args()

    caption_path = args.results_save_path
    save_path = os.path.join(os.path.dirname(caption_path), f"{os.path.basename(caption_path)}_scores")
    os.makedirs(save_path, exist_ok=True)  
    
    human_annotations = pd.read_csv(args.human_annotations_path)

    minilm_model = SentenceTransformer("all-MiniLM-L6-v2")

    uids = os.listdir(caption_path)
    uids = [uid.split('.')[0] for uid in uids]

    for uid in uids:
        selected_row = human_annotations[human_annotations['folder_name'] == uid]
        if len(selected_row) == 0:  # 检查是否找到对应的行
            print(f"Warning: No data found for uid {uid}")
            continue
            

        read_json_path = os.path.join(caption_path, f"{uid}.json")
        write_json_path = os.path.join(save_path, f"{uid}.json")
        
        if not os.path.exists(read_json_path):  # 检查输入文件是否存在
            print(f"Warning: Input file not found: {read_json_path}")
            continue
            
        try:
            with open(read_json_path, 'r') as f:
                read_json_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON file: {read_json_path}")
            continue

        caption_names = read_json_data.keys()

        write_json_data = {}
        if os.path.exists(write_json_path):
            with open(write_json_path, 'r') as f:
                write_json_data = json.load(f)

        for caption_name in caption_names:


            try:
                if "human" in caption_name:
                    human_idx = read_json_data[caption_name]["caption_idx"]
                    caption = read_json_data[caption_name]["human_caption"]
                    available_indices = list(range(1, 6))
                    available_indices.remove(human_idx)
                    gt_captions = [selected_row[f'caption_{id}'].values[0] for id in available_indices]
                else:
                    caption = read_json_data[caption_name]
                    gt_captions = [selected_row[f'caption_{id}'].values[0] for id in range(1, 6)]




                # BERT-score
                P, R, F1 = bert_score([caption], [gt_captions], model_type="bert-base-uncased", lang="en")
                bert_f1 = F1.mean().item()
                print(f"bert_score (F1): {bert_f1}")
                write_json_data[f"{caption_name}_BERT_score"] = bert_f1

                # MiniLM Score
                minilm_score = float(torch.max(minilm_model.similarity(
                    minilm_model.encode([caption]), minilm_model.encode(gt_captions)
                )).numpy())
                print(f"minilm_score: {minilm_score}")
                write_json_data[f"{caption_name}_MiniLM_score"] = minilm_score

                # Compute other scores
                cider_score, bleu_score, meteor_score, rouge_score = compute_scores(gt_captions, caption)

                print(f"CIDEr: {cider_score}")
                print(f"BLEU: {bleu_score}")
                print(f"METEOR: {meteor_score}")
                print(f"ROUGE: {rouge_score}")

                write_json_data[f"{caption_name}_CIDEr_score"] = cider_score
                for i in range(1, 5):
                    write_json_data[f"{caption_name}_BLEU_{i}_score"] = bleu_score[i - 1]
                write_json_data[f"{caption_name}_METEOR_score"] = meteor_score
                write_json_data[f"{caption_name}_ROUGE_score"] = rouge_score

            except Exception as e:
                print(f"Error processing caption {caption_name}: {str(e)}")
                continue

        with open(write_json_path, 'w') as json_file:
            json.dump(write_json_data, json_file, indent=4)
