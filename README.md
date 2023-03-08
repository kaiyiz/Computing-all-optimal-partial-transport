# Computing All Optimal Partial Transport

This repository contains the experiment code for the paper "Computing All Optimal Partial Transports", which is available on [OpenReview](https://openreview.net/forum?id=gwcQajoXNF).

The repository is divided into two parts:
<ol>
    <li>Implementation of optimal partial transport in outlier detection</li>
    <li>Implementation of optimal partial transport in PU-learning</li>
</ol>

For the outlier detection part, we compare our results with [Mukherjee et al. (2021)](https://arxiv.org/abs/2012.07363), whose code was obtained from their Github repository https://github.com/debarghya-mukherjee/Robust-Optimal-Transport, and some of their code was adopted for our comparison.

The scripts named "outlier_detect_minist.py" and "outlier_detect_sythetic.py" generate the performance comparison in table 1 of our paper, conducting outlier detection with MNIST and synthetic data, respectively.

For PU-learning, we compare our results with [Chapel et al. (2020)](https://arxiv.org/abs/2002.08276), and some of their code was obtained from their Github repository https://github.com/lchapel/partial-GW-for-PU and adopted for our comparison.

The notebook named "table2.ipynb" conducts the experiments of PU-learning and generates the performance results in table 2 of the paper.

## Dependencies

This code is written in Python 3 and relies on the following libraries:
```
numpy
scipy
pot
tensorflow
jpype
kneed
```

## Citation

If you find this work helpful for your research, please consider citing our paper:
```
@inproceedings{APhatak2023,
title = {Computing all Optimal Partial Transports},
author = {Abhijeet Phatak and Sharath Raghvendra and Chittaranjan Tripathy and Kaiyi Zhang},
year = {2023},
booktitle={International Conference on Learning Representations (ICLR)},
url={https://openreview.net/forum?id=gwcQajoXNF}
}
```