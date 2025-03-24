# DATTA: Domain Diversity Aware Test-Time Adaptation for Dynamic Domain Shift Stream

This repository is the official implementation of

**[DATTA: Domain Diversity Aware Test-Time Adaptation for Dynamic Domain Shift Stream](https://arxiv.org/abs/2408.08056), ICME, 2025**\
Chuyang Ye†, Dongyan Wei†, Zhendong Liu, Yuanyi Pang, Yixi Lin, Qinting Jiang, Jingyan Jiang*, Dongbiao He

It's based on the  **[TTAB repository](https://github.com/LINs-lab/ttab)**, which is offically implementation of the paper **[On Pitfalls of Test-time Adaptation](https://arxiv.org/abs/2306.03536)**.


**Introduction**  

Test-Time Adaptation (TTA) addresses domain shifts between training and testing data to maintain model robustness in real-world scenarios. However, existing TTA methods assume a static target domain (e.g., homogeneous single-domain data) at each moment, failing to handle **dynamic domain shift streams**—where data distributions unpredictably alternate between single-domain and multi-domain configurations. Such dynamics, common in applications like autonomous driving (e.g., abrupt shifts from clear daytime to rainy urban night scenes), cause performance degradation due to inaccurate batch normalization (BN) statistics and gradient conflicts during adaptation.  

This paper introduces **DATTA**, the **first TTA framework** explicitly designed to address **dynamic domain shift streams**. Its key contributions are **three pioneering innovations**, **first proposed** to tackle evolving domain diversity in real-world data streams:  
1. **Domain-Diversity Score**: **The first metric** for quantifying real-time domain transitions in **dynamic domain shift streams**, evaluating alignment between samples and batch distributions via BN statistics and feature maps. This capability, absent in prior static TTA methods, enables proactive adaptation to unpredictable domain mixtures.  
2. **Domain-Diversity Adaptive BN (DABN)**: **The first normalization mechanism** tailored for **dynamic domain shifts**, dynamically blending source and test-time statistics based on diversity scores. DABN overcomes the instability of fixed BN strategies under rapid domain alternations.  
3. **Domain-Diversity Adaptive Fine-Tuning (DAFT)**: **The first optimization strategy** designed for multi-domain mixtures in streaming data, selectively updating parameters to mitigate gradient conflicts while maintaining robustness to transient domains.  

Experiments demonstrate DATTA’s superiority over state-of-the-art methods (by up to **13%**) under **dynamic domain shift streams**, validating its effectiveness in real-world scenarios with evolving domain diversity. 


<table>
  <tr>
    <td><img src="./figs/overview_of_scenarios.png" width="100%"></td>
    <td><img src="./figs/overview_of_method.png" width="115%"></td>
  </tr>
</table>

### Requirements
```
conda env create -f environment.yml
```
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

### Model
Our implementations use [ResNet-50](https://drive.google.com/file/d/1-qUXRp4iwq_Q28NfyFQIWXPwlZAyYVPB/view?usp=sharing) (He et al., 2015) and [EfficientVit-M5](https://github.com/mit-han-lab/efficientvit) (Cai et al., 2024).

## Using the example scripts
We provide an example script that can be used to adapt distribution shifts on the TTAB datasets. 

```bash
python run_exp.py
```

For more details, please refer to **[TTAB repository](https://github.com/LINs-lab/ttab)**.




