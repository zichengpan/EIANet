---

<div align="center">    
 
# EIANet: A Novel Domain Adaptation Approach to Maximize Class Distinction with Neural Collapse Principles

[Zicheng Pan](https://zichengpan.github.io/), Xiaohan Yu, and Yongsheng Gao

[![Paper](https://img.shields.io/badge/paper-BMVC%202024-4B275F.svg)](https://bmvc2024.org/proceedings/317/)

</div>

## Abstract
Source-free domain adaptation (SFDA) aims to transfer knowledge from a labelled source domain to an unlabelled target domain. A major challenge in SFDA is deriving accurate categorical information for the target domain, especially when sample embeddings from different classes appear similar. This issue is particularly pronounced in fine-grained visual categorization tasks, where inter-class differences are subtle. To overcome this challenge, we introduce a novel ETF-Informed Attention Network (EIANet) to separate class prototypes by utilizing attention and neural collapse principles. More specifically, EIANet employs a simplex Equiangular Tight Frame (ETF) classifier in conjunction with an attention mechanism, facilitating the model to focus on discriminative features and ensuring maximum class prototype separation. This innovative approach effectively enlarges the feature difference between different classes in the latent space by locating salient regions, thereby preventing the misclassification of similar but distinct category samples and providing more accurate categorical information to guide the fine-tuning process on the target domain. Experimental results across four SFDA datasets validate EIANet's state-of-the-art performance.

## Dataset Preparation Guideline
Please prepare the datasets according to the configuration files in the data folder. You may refer to these sites to download the datasets: [Office-31 / Office-Home](https://github.com/tim-learn/SHOT), [CUB-200-Paintings](https://github.com/thuml/PAN), [Birds31](https://drive.google.com/file/d/1vDsSf3oAo5OQQdk8asveH5VYu76ds1XR/view?usp=sharing). The default data structure is given as follows:

```
./data
|–– birds31/
|–– cub/
|–– office/
|–– office-home/
|   |–– domain1/
|   |–– domain2/
|   |-- ...
|   |–– domain1.txt
|   |-- domain2.txt
|   |-- ...
```

## Citation
If you find our code or paper useful, please give us a citation, thanks!
```
@inproceedings{Pan_2024_BMVC,
author    = {Zicheng Pan and Xiaohan Yu and Yongsheng Gao},
title     = {EIANet: A Novel Domain Adaptation Approach to Maximize Class Distinction with Neural Collapse Principles},
booktitle = {35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow, UK, November 25-28, 2024},
publisher = {BMVA},
year      = {2024},
url       = {https://papers.bmvc2024.org/0317.pdf}
}
```

## Acknowledgment
We thank the following repos for providing helpful components/functions in our work.

- [SHOT](https://github.com/tim-learn/SHOT)

- [AaD](https://github.com/Albert0147/AaD_SFDA)

- [PAN](https://github.com/thuml/PAN)
