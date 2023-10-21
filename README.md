# CPATT
Prompt-based Event Relation Identification with Constrained Prefix ATTention Mechanism

We propose a prompt-based method to identify the causal  and temporal relation and employ a Constrained Prefix ATTention (CPATT) mechanism to ameliorate the prompt-tuning process.

# Introduction of Code
We provide two different sets of codes for different datasets and experimental settings.  

Each set of codes corresponds to a different experimental setup, but there are no structural changes.  

Taking the Event Causality Identification as an example:  

makedataset_eventstoryline.py corresponds to the template creation module.  

train.py corresponds to the prompt-tunning module.  

inference.py corresponds to the relation inference module.  

# Introduction of Data
Three different datasets are provided in the this folder.  

# BibTex
```
@article{ZHANG2023111072,
title = {Prompt-based event relation identification with Constrained Prefix ATTention mechanism},
journal = {Knowledge-Based Systems},
volume = {281},
pages = {111072},
year = {2023},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2023.111072},
url = {https://www.sciencedirect.com/science/article/pii/S0950705123008225},
author = {Hang Zhang and Wenjun Ke and Jianwei Zhang and Zhizhao Luo and Hewen Ma and Zhen Luan and Peng Wang},
}
```
