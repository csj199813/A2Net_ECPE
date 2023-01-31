# A2Net_ECPE

## Joint Alignment of Multi-Task Feature and Label Spaces for Emotion Cause Pair Extraction
This repository contains the code of the official implementation for the paper: Joint Alignment of Multi-Task Feature and Label Spaces for Emotion Cause Pair Extraction. The paper has been accepted to appear at Coling 2022.

Some code is based on [Rank-Emotion-Cause](https://github.com/Determined22/Rank-Emotion-Cause), and [Partition Filter Network](https://github.com/Coopercoppers/PFN).

## Requirements
* CUDA：11.4
* Python 3
* PyTorch 1.10.2

## Quick Start
1. Download the pertrained ["BERT-Base, Chinese"](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz) model. And then put the model file pytorch_model.bin to the folder src/bert-base-chinese

2. python src/main.py
