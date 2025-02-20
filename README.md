# DATTA: Domain Diversity Aware Test-Time Adaptation for Dynamic Domain Shift Stream

This repository is the official implementation of
<br>
**[DATTA: Domain Diversity Aware Test-Time Adaptation for Dynamic Domain Shift Stream]()**
<br> 
It's based on the  **[TTAB repository](https://github.com/LINs-lab/ttab)**, which is offically implementation of the paper **[On Pitfalls of Test-time Adaptation](https://arxiv.org/abs/2306.03536)** 


## Overview
Our repository contains:
1. The implementation of **[ViDA](https://arxiv.org/abs/2306.04344)**, **[DeYO](https://arxiv.org/abs/2403.07366)**, and **our method** on TTAB
2. The code introducing new scenario.


## News

### Requirements
The TTAB package depends on the following requirements:
- finch-clust>=0.2.0
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

For more details, please refer to **[TTAB repository](https://github.com/LINs-lab/ttab)**.




