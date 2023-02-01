# A2Net_ECPE

## Joint Alignment of Multi-Task Feature and Label Spaces for Emotion Cause Pair Extraction
This repository contains the code of the official implementation for the paper: Joint Alignment of Multi-Task Feature and Label Spaces for Emotion Cause Pair Extraction. The paper has been accepted to appear at Coling 2022.

Some code is based on [Rank-Emotion-Cause](https://github.com/Determined22/Rank-Emotion-Cause), and [Partition Filter Network](https://github.com/Coopercoppers/PFN).

If you use our codes or your research is related to our paper, please kindly cite our paper:

```
@inproceedings{chen-etal-2022-joint,
    title = "Joint Alignment of Multi-Task Feature and Label Spaces for Emotion Cause Pair Extraction",
    author = "Chen, Shunjie  and
      Shi, Xiaochuan  and
      Li, Jingye  and
      Wu, Shengqiong  and
      Fei, Hao  and
      Li, Fei  and
      Ji, Donghong",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.606",
    pages = "6955--6965",
}

```

## Requirements
* CUDA：11.4
* Python 3
* PyTorch 1.10.2

The code has been tested on Ubuntu 20.04.3 LTS using a single 3090(24G).

## Quick Start
1. Download the pertrained ["BERT-Base, Chinese"](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz) model. And then put the model file `pytorch_model.bin` to the folder `src/bert-base-chinese`

2. python src/main.py
