# 4D Object Question Answering
## Folder Structure
```
4D_Object_Question_Answering
├── data
│   ├── 4d_qa.json （4D QA data）
│   └── 4d_object_multi_view_videos （4D object multi-view videos）
├── README.md
├── eval_code_example (code to reproduce the results in the paper)
│   ├── gemini-1.5-flash_exp.py
│   ├── qwen2_vl_7b_exp.py
│   └── ....
```
## How to Run
```
# gemini-1.5-flash
python gemini-15-flash_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --api_key <your_gemini_api_key> 

# gemini-1.5-pro
python gemini-15-pro_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --api_key <your_gemini_api_key> 

# gpt-4o
python gpt-4o_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --api_key <your_gpt_api_key> 

# gpt-4o-mini
python gpt-4o-mini_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --api_key <your_gpt_api_key> 

# qwen2_vl_7b
python qwen2_vl_7b_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --cache_dir <your_model_weights_path>

# qwen2_vl_72b
python qwen2_vl_72b_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --cache_dir <your_model_weights_path>

# llava-video 7b
python ./llava_video_7b_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --cache_dir <your_model_weights_path>

# llava-video 72b
python ./llava_video_72b_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --cache_dir <your_model_weights_path>

# llava-onevision 7b
python ./llava_onevision_7b_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --cache_dir <your_model_weights_path>

# llava-onevision 72b
python ./llava_onevision_72b_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --cache_dir <your_model_weights_path>

# Internvl2 8b
python ./InternVL2_8b_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --cache_dir <your_model_weights_path>

# Internvl2 76b
python ./InternVL2_76b_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --cache_dir <your_model_weights_path>

# videochat2 
# There is no need to set the cache_dir, but you should set the "videochat2_model_path_stage3", "vit_blip_model_path" and "videochat2_model_path_stage3" in the video_chat2/configs/config_mistral.json
# For more details, please refer to https://github.com/OpenGVLab/Ask-Anything
python videochat2_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> 

# minigpt4-video
# There is no need to set the cache_dir, but you need to set "ckpt", you should also set the "ckpt" in the minigpt4/text_configs/mistral_test_config.yaml
# For more details, please refer to https://github.com/Vision-CAIR/MiniGPT4-video
python minigpt4-video_exp.py --save_path <your_save_path> --vqa_file_path <your_vqa_file_path> --video_data_path <your_video_data_path> --ckpt <video_mistral_checkpoint_last.pth file path>
```


