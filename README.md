# Is Extending Modality The Right Path Towards Omni-Modality?

<!-- ![Static Badge](https://img.shields.io/badge/vision%20and%20language-blue)
![Static Badge](https://img.shields.io/badge/knowledge%20conflict-blue) -->

Code and data for paper [Is Extending Modality The Right Path Towards Omni-Modality?](https://arxiv.org/abs/xxxx.xxxxx).

<p align="center">
    [<a href="https://darthzhu.github.io/lm-extend-page/">Website</a>] •
    [<a href="https://arxiv.org/abs/xxxx.xxxxx">Paper</a>]
    <!-- [<a href="https://huggingface.co/datasets/DarthZhu/vlm-knowledge-conflict">Dataset</a>] • -->
    <!-- [<a href="https://x.com/_vztu/status/1843350510583374306">Twitter</a>] -->
</p>


## Environment Setup

To install the inference environment, run the following code:

```bash
conda env create -f environment.yml
```

## Inference

To generate answers from LLMs, run the script `scripts/infer.sh`.
The script has three steps:
1. Download the model tensors from huggingface.
2. Extract the LLM component from the multimodal model.
3. Generate.

In case certain steps are not required for specific models, you can delete these steps on your own.

To generate answers from multimodal models, run the script `scripts/infer_multimodal.sh`.

For merged models,  you need to first run `python src/utils/save_merged_vlm.py` to load the merged LLM into the multimodal model.
You can change the target model name in the python file.

## Fine-tuning

To train the merged model, you can run the script `src/training/Qwen2.5-VL/qwen-vl-finetune/train.sh` adapted from [Qwen2.5-VL training codes](https://github.com/QwenLM/Qwen2.5-VL).

## Citation

If you find this repo useful, please cite the following paper:

```bib
@article{zhu2025extend,
  title={Is Extending Modality The Right Path Towards Omni-Modality?},
  author={Zhu, Tinghui and Zhang, Kai and Chen, Muhao and Su, Yu},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```