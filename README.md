<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding</h1>      
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="#" target="_blank" style="text-decoration: none;">Wenxuan Zhu</a><sup>1*</sup>,&nbsp;
    <a href="#" target="_blank" style="text-decoration: none;">Bing Li</a><sup>1*</sup>,&nbsp;
    <a href="#" target="_blank" style="text-decoration: none;">Cheng Zheng</a><sup>1*</sup>,&nbsp;
    <a href="#" target="_blank" style="text-decoration: none;">Jinjie Mai</a><sup>1</sup>,&nbsp;
    <a href="#"  target="_blank" style="text-decoration: none;">Jun Chen</a><sup>1</sup>,&nbsp;
    <a href="#" target="_blank" style="text-decoration: none;">Letian Jiang</a><sup>1</sup>,&nbsp;
    <a href="#" target="_blank" style="text-decoration: none;">Abdullah Hamdi</a><sup>2</sup>,&nbsp;<br/>
    <a href="#" target="_blank" style="text-decoration: none;">Sara Rojas Martinez</a><sup>1</sup>,&nbsp;
    <a href="#"  target="_blank" style="text-decoration: none;">Chia-Wen Lin</a><sup>3</sup>,&nbsp;
    <a href="#"  target="_blank" style="text-decoration: none;">Mohamed Elhoseiny</a><sup>1</sup>,&nbsp;
    <a href="#"  target="_blank" style="text-decoration: none;">Bernard Ghanem</a><sup>1</sup><br/>
&nbsp;<sup>1</sup>KAUST,&nbsp;<sup>2</sup>University of Oxford,&nbsp;<sup>3</sup>National Tsing Hua University<br/>
<a href="https://wenxuanzhu1103.github.io/4dbench.github.io/" title="Website" target="_blank" rel="nofollow" style="text-decoration: none;">🌎Website</a> |
<a href="https://huggingface.co/datasets/vxuanz/4D-Bench" title="Dataset" target="_blank" rel="nofollow" style="text-decoration: none;">🤗 Dataset</a> |
<a href="https://arxiv.org/abs/2503.17827" title="aXiv" target="_blank" rel="nofollow" style="text-decoration: none;">📄 arXiv</a> |
</p>


# Abstract
Multimodal Large Language Models (MLLMs) have demonstrated impressive 2D image/video understanding capabilities.
However, there are no publicly standardized benchmarks to assess the abilities of MLLMs in understanding the 4D objects (3D objects with temporal evolution over time).
In this paper, we introduce 4D-Bench, the first benchmark to evaluate the capabilities of MLLMs in 4D object understanding, featuring tasks in 4D object Question Answering (4D object QA) and 4D object captioning.
4D-Bench provides 4D objects with diverse categories, high-quality annotations, and tasks necessitating multi-view spatial-temporal understanding, different from existing 2D image/video-based benchmarks.
With 4D-Bench, we evaluate a wide range of open-source and closed-source MLLMs.
The results from the 4D object captioning experiment indicate that MLLMs generally exhibit weaker temporal understanding compared to their appearance understanding, notably, while open-source models approach closed-source performance in appearance understanding, they show larger performance gaps in temporal understanding.
4D object QA yields surprising findings: even with simple single-object videos, MLLMs perform poorly, with state-of-the-art GPT-4o achieving only 63\% accuracy compared to the human baseline of 91\%.
These findings highlight a substantial gap in 4D object understanding and the need for further advancements in MLLMs.
# How to run
1. Download dataset from huggingface and unzip: https://huggingface.co/datasets/vxuanz/4D-Bench
2. Run the evaluation code following the instructions in the README.md files under the 4D_Object_Captioning and 4D_Object_Question_Answering directories.

# Results

## 4D Object Captioning Results

| **Model** | **CIDEr** | **BLEU@4** | **METEOR** | **ROUGE** | **BERT** | **SBERT** | **GPT-Appearance** | **GPT-Action** | **GPT-Eval** |
|:----------|:---------:|:----------:|:----------:|:---------:|:--------:|:---------:|:------------------:|:--------------:|:------------:|
| MiniGPT4-Video | 18.4 | 0.6 | 23.1 | 13.2 | 50.7 | 51.2 | 1.737/5 | 1.351/5 | 1.544/5 |
| InternVL2 8B | 48.4 | 2.5 | 27.9 | 22.6 | 58.2 | 60.3 | 2.531/5 | 1.877/5 | 2.204/5 |
| VideoChat2-Mistral | 79.0 | 6.9 | 33.5 | 33.5 | 65.4 | 59.7 | 2.578/5 | 1.912/5 | 2.245/5 |
| LLaVA-OneVison 7B | 86.4 | 10.0 | 39.2 | 32.7 | 63.2 | 65.6 | 3.166/5 | 2.479/5 | 2.823/5 |
| LLaVA-Video 7B | 102.6 | 14.6 | **41.7** | 38.8 | 66.7 | 68.1 | 3.235/5 | 2.552/5 | 2.894/5 |
| Qwen2-VL 7B | 84.5 | 10.1 | 36.9 | 36.4 | 65.7 | 66.9 | 3.170/5 | 2.666/5 | 2.918/5 |
| InternVL2 76B | 72.0 | 5.5 | 34.2 | 27.1 | 60.9 | 65.3 | 3.099/5 | 2.637/5 | 2.868/5 |
| LLaVA-OneVision 72B | **107.4** | **16.1** | 41.1 | **41.5** | **68.5** | 68.0 | 3.180/5 | 2.268/5 | 2.724/5 |
| LLaVA-Video 72B | 106.2 | 15.1 | 39.8 | 40.9 | **68.5** | 68.1 | 3.138/5 | 2.471/5 | 2.804/5 |
| Qwen2-VL 72B | 95.1 | 12.4 | 40.3 | 38.0 | 66.8 | 67.5 | 3.324/5 | 2.791/5 | 3.057/5 |
| Gemini 1.5 Flash | 84.3 | 7.3 | 36.5 | 32.9 | 65.3 | **68.9** | 3.246/5 | 2.931/5 | 3.088/5 |
| GPT-4o mini | 51.1 | 2.7 | 30.8 | 24.0 | 59.3 | 63.5 | *3.311/5* | *3.131/5* | *3.221/5* |
| Gemini 1.5 Pro | 94.8 | 11.2 | 38.7 | 39.0 | **68.5** | 68.8 | 3.311/5 | 2.983/5 | 3.147/5 |
| GPT-4o | 69.0 | 6.4 | 35.9 | 32.1 | 64.1 | 66.4 | ***3.507/5*** | ***3.258/5*** | ***3.382/5*** |
| Average | - | - | - | - | - | - | 3.038/5 | 2.522/5 | 2.780/5 |
| **Human** | 126.6 | 14.12 | 45.01 | 43.48 | 71.69 | 76.30 | 3.772/5 | 3.879/5 | 3.826/5 |

<!-- *The Average row represents the mean performance of all tested MLLM models under each metric. The Human row represents the performance of human annotator under each metric. For each metric, we bold the best performing MLLM model. We highlight GPT metrics as they demonstrate better alignment with human preferences in evaluating caption quality, and our analysis also primarily focuses on models' performance across these metrics. GPT-4o's GPT metrics are marked in gray due to the potential self-evaluation bias when using GPT-based metrics to evaluate a GPT model. We provide human performance as a reference.* -->

## 4D Object Question Answering Results

| **Model** | **Object Counting (%)** | **Temporal Relationship (%)** | **Action (%)** | **Spatial Relationship (%)** | **Appearance (%)** | **Overall (%)** |
|:----------|:-----------------------:|:-----------------------------:|:--------------:|:----------------------------:|:------------------:|:---------------:|
| MiniGPT4-Video | 22.05 | 26.43 | 22.90 | 22.39 | 22.06 | 23.17 |
| VideoChat2 | 22.83 | 31.43 | 33.18 | 38.81 | 34.56 | 32.36 |
| InternVL2 8B | 18.11 | 31.43 | 35.98 | 32.09 | 39.71 | 32.09 |
| LLaVA-OneVision 7B | 42.52 | 52.86 | 42.99 | 57.46 | 74.26 | 53.00 |
| LLaVA-Video 7B | 42.52 | 55.00 | 52.80 | 56.72 | **78.68** | 56.86 |
| Qwen2-VL 7B | 38.58 | 56.43 | 57.94 | 58.96 | 71.32 | 56.99 |
| InternVL2 76B | 28.35 | 45.00 | 42.52 | 38.81 | 64.71 | 43.94 |
| LLaVA-OneVision 72B | 49.61 | 58.57 | 60.75 | 61.19 | 76.47 | 61.38 |
| LLaVA-Video 72B | **54.33** | 58.57 | 57.48 | 66.42 | 77.21 | 62.32 |
| Qwen2-VL 72B | 45.67 | 55.71 | 58.41 | 61.19 | 72.06 | 58.72 |
| Gemini 1.5 Flash | 26.77 | 50.00 | 53.27 | 60.45 | 66.18 | 51.80 |
| GPT-4o mini | 40.16 | 50.71 | 50.00 | 61.94 | 72.06 | 54.59 |
| Gemini 1.5 Pro | 46.46 | 58.57 | 59.35 | 64.18 | 68.38 | 59.52 |
| GPT-4o | 44.09 | **59.29** | **63.55** | **69.40** | 77.21 | **62.98** |
| Average | 37.29 | 49.29 | 49.37 | 53.57 | 63.92 | 50.69 |
| **Human** | 88.98 | 89.29 | 94.39 | 91.04 | 89.71 | 91.08 |

<!-- *The Overall column refers to average accuracy across all sub-tasks. The Average row represents the mean performance of all tested models in each category. We provide human performance as a reference.* -->


# Benchmark Data Usage Restrictions
4D-Bench is strictly for academic research purposes, and any form of commercial use is prohibited. The copyright of all 4D objects is retained by their respective owners, and proper acknowledgement will be given in the dataset. The dataset as a whole is licensed under the ODC-By v1.0 license, consistent with the licensing of Objaverse-XL
# Acknowledgment
This project  is inspired by [Objaverser-XL](https://objaverse.allenai.org/) and [Video-MME](https://video-mme.github.io/home_page.html). 
# Citation
If you find this projects is useful, please cite:
```tex
@misc{zhu20254dbenchbenchmarkingmultimodallarge,
      title={4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding}, 
      author={Wenxuan Zhu and Bing Li and Cheng Zheng and Jinjie Mai and Jun Chen and Letian Jiang and Abdullah Hamdi and Sara Rojas Martinez and Chia-Wen Lin and Mohamed Elhoseiny and Bernard Ghanem},
      year={2025},
      eprint={2503.17827},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.17827}, 
}
```