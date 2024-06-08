<<<<<<< HEAD
# Discover Your Neighbors(DYN): Advanced Stable Test-Time Adaptation in Dynamic World

This repository is the official implementation of
<br>
**[Discover Your Neighbors: Advanced Stable Test-Time Adaptation in Dynamic World]()**
<br> 
It's based on the TTAB repository **[On Pitfalls of Test-time Adaptation](https://github.com/LINs-lab/ttab)**


> TL;DR: We introduce a test-time adaptation benchmark that systematically examines a large array of recent methods under diverse conditions. 
> - Model selection is exceedingly difficult for test-time adaptation due to online batch dependency.
> - The effectiveness of TTA methods varies greatly depending on the quality and properties of pre-trained models.
> - Even with oracle-based tuning, no existing methods can yet address all common classes of distribution shifts.


## Overview

Our repository contains:
1. dyn, vida, deyo
2. 新场景


## News

- June 2024: We updated the implementation of our Test-time Adaptation method.



## Installation
To run a baseline test, please prepare the relevant pre-trained checkpoints for the base model and place them in `pretrain/ckpt/`.
### Requirements
The TTAB package depends on the following requirements:

- numpy>=1.21.5
- pandas>=1.1.5
- pillow>=9.0.1
- pytz>=2021.3
- torch>=1.7.1
- torchvision>=0.8.2
- timm>=0.6.11
- scikit-learn>=1.0.3
- scipy>=1.7.3
- tqdm>=4.56.2
- tyro>=0.5.5



## Using the example scripts
We provide an example script that can be used to adapt distribution shifts on the TTAB datasets. 

```bash
python run_exp.py
```





