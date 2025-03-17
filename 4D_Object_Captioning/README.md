# 4D Object Captioning
## Folder Structure
```
4D_Object_Captioning
├── data
│   ├── 4d_object_multi_view_videos （4D object multi-view videos）
│   └── human_annotations.csv （human caption data）
├── README.md
├── code
│   ├── mllm_gen_caption
│   │   ├── gemini-15-flash_caption_exp.py
│   │   ├── gemini-15-pro_caption_exp.py
│   │   ├── gpt-4o_caption_exp.py
│   │   └── ....
│   ├── eval_metrics
│   │   ├── llm_score (GPT-Appearance and GPT-Action metrics)
│   │   └── other_metrics (BLEU, CIDEr, METEOR, ROUGE-L, BERTScore, SBERTScore metrics)
```
## MLLM generate caption
```
# gemini-1.5-flash
python gemini-15-flash_caption_exp.py --api_key <your_gemini_api_key> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>

# gemini-1.5-pro
python gemini-15-pro_caption_exp.py --api_key <your_gemini_api_key> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>

# gpt-4o
python gpt-4o_caption_exp.py --api_key <your_gpt_api_key> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>salloc --cpus-per-task=10 --account conf-icml-2025.01.31-ghanembs   --gres=gpu:a100:1 --mem=64GB --time=04:00:00

# gpt-4o-mini
python gpt-4o-mini_caption_exp.py --api_key <your_gpt_api_key> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>

# qwen2_vl_7b
python qwen2_vl_7b_exp.py --cache_dir <your_model_weights_path> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>

# qwen2_vl_72b
python qwen2_vl_72b_exp.py --cache_dir <your_model_weights_path> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>

# llava-video 7b
python ./llava_video_7b_exp.py --cache_dir <your_model_weights_path> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>

# llava-video 72b
python ./llava_video_72b_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --cache_dir <your_model_weights_path>

# llava-onevision 7b
python ./llava_onevision_7b_exp.py --cache_dir <your_model_weights_path> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>

# llava-onevision 72b
python ./llava_onevision_72b_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --cache_dir <your_model_weights_path>

# Internvl2 8b
python ./InternVL2_8b_exp.py --cache_dir <your_model_weights_path> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>

# Internvl2 76b
python ./InternVL2_76b_exp.py --cache_dir <your_model_weights_path> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>

# videochat2 
# There is no need to set the cache_dir, but you should set the "videochat2_model_path_stage3", "vit_blip_model_path" and "videochat2_model_path_stage3" in the video_chat2/configs/config_mistral.json
# For more details, please refer to https://github.com/OpenGVLab/Ask-Anything
python videochat2_exp.py --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>

# minigpt4-video
# There is no need to set the cache_dir, but you need to set "ckpt", you should also set the "ckpt" in the minigpt4/text_configs/mistral_test_config.yaml
# For more details, please refer to https://github.com/Vision-CAIR/MiniGPT4-video
python minigpt4-video_exp.py --ckpt <video_mistral_checkpoint_last.pth file_path> --video_data_path <your_video_data_path>  --results_save_path <your_results_save_path>
```
## Eval metrics
```
# GPT-Appearance and GPT-Action metrics
python llm_score/gpt_appearance_action_metrics.py --results_save_path <your_results_save_path>

# BLEU, CIDEr, METEOR, ROUGE-L, BERTScore, SBERTScore metrics
python other_metrics/eval_metrics.py --results_save_path <your_results_save_path>

```