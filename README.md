# Discover Your Neighbors(DYN): Advanced Stable Test-Time Adaptation in Dynamic World

This repository is the official implementation of
<br>
**[Discover Your Neighbors: Advanced Stable Test-Time Adaptation in Dynamic World]()**
<br> 
It's based on the  **[TTAB repository](https://github.com/LINs-lab/ttab)**, which is offically implementation of the paper **[On Pitfalls of Test-time Adaptation](https://arxiv.org/abs/2306.03536)** 


> We introduce a novel test-time adaptation method DYN whose core innovation is identifying similar samples via instance normalization statistics and clustering into groups which provides consistent class-irrelevant representations.
> - To our knowledge, this is the first backward-free TTA method addressing dynamic test patterns, and we conduct sufficient measurements to re-understanding of the batch normalization statistics through class-related and class-irrelevant features. 
> - We propose a test-time normalization approach that utilizes instance normalization statistics to cluster samples with similar category-independent distributions. Combining TCN and SBN statistics enables robust representations adaptable to dynamic data.
> - Experiments on benchmark datasets demonstrate robust performance compared to state-of-the-art studies under dynamic distribution shifts, with up to a 35\% increase in accuracy.

## Overview
Our repository contains:
1. dyn, vida, deyo
2. 新场景


## News
- June 2024: We updated the implementation of our Test-time Adaptation method DYN(Discover Your Neighbors).

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





