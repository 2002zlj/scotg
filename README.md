# SCOTG

This repository contains implementation code for SCOTG.

## Environment 
NVIDIA GeForce RTX 3090

Ubuntu 20.04.6 LTS

CUDA 11.8

Firstly, install the environments
```
conda env create -f environment.yml
cd DualPrompt
conda env create -f environment.yml
cd Privilege
conda env create -f environment.yml
```

## Dataset and model weights
Download the dataset and replace all places in the code that use the dataset path with your own data path.

The dataset is <a href="https://pan.baidu.com/s/11RrKBcM08elbQu8K9vQU7Q">here</a> and the password is nrut

Download the model weights for each method and place them in their respective folders

The model weights are <a href="https://pan.baidu.com/s/13tln4r4XLClqzlorPhAWww">here</a> and the password is 1l4b

## Experiment
```
# SCOTG
conda activate scotg
python test_SCOTG.py

# CPE-CLIP、ZSCL、CLIP finetune、CLIP zeroshot and ablation 
conda activate scotg
bash test.sh

# L2P
cd L2P
conda activate l2p
python test_l2p.py

# DualPrompt
cd DualPrompt
conda activate l2p
python test_dualprompt.py

# Privilege
cd Privilege
conda activate privilege
python test.py
```

## Acknowledgment

Our project is based on 
- [l2p-pytorch](https://github.com/JH-LEE-KR/l2p-pytorch) 
- [dualprompt-pytorch](https://github.com/JH-LEE-KR/dualprompt-pytorch) 
- [PriViLege](https://github.com/KU-VGI/PriViLege)
- [cpe-clip](https://github.com/neuraptic/cpe-clip) 
We sincerely thanks to offering useful public code base.

## Note
This repository is only used for academic.
