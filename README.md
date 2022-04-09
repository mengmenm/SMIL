# SMIL: Multimodal Learning with Severely Missing Modality

Pytorch implementation of [SMIL: Multimodal Learning with Severely Missing Modality](https://arxiv.org/pdf/2103.05677.pdf) by Mengmeng Ma and Xi Peng.

## Abstract

A common assumption in multimodal learning is the completeness of training data, i.e., full modalities are available in all training examples. Although there exists research endeavor in developing novel methods to tackle the incompleteness of testing data, e.g., modalities are partially missing in testing examples, few of them can handle incomplete training modalities. The problem becomes even more challenging if considering the case of severely missing, e.g., 90% training examples may have incomplete modalities. For the first time in the literature, this paper formally studies multimodal learning with missing modality in terms of flexibility (missing modalities in training, testing, or both) and efficiency (most training data have incomplete modality). Technically, we propose a new method named SMIL that leverages Bayesian meta-learning in uniformly achieving both objectives. To validate our idea, we conduct a series of experiments on three popular benchmarks: MM-IMDb, CMU-MOSI, and avMNIST. The results prove the state-of-the-art performance of SMIL over existing methods and generative baselines including autoencoders and generative adversarial networks.

## Requirements

```bash
pip install -r requirements.txt
```

# Datasets

#### [AV-MNIST](https://arxiv.org/abs/1808.07275) dataset

- Download the data [here](https://drive.google.com/file/d/1JTS--8d_BxzZfhQfSAAYeYTjCdUbJyuD/view?usp=sharing) and extracted it to /data

## Training

Train modal-specific baselines

```bash
bash run_experiment.sh
```

To do

- [ ] Training with missing modality

## Citation

If you find our code useful in your research, please consider citing:

```
@inproceedings{ma2021smil,
  title={SMIL: Multimodal Learning with Severely Missing Modality},
  author={Ma, Mengmeng and Ren, Jian and Zhao, Long and Tulyakov, Sergey and Wu, Cathy and Peng, Xi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2302--2310},
  year={2021}
}
```

