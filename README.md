## Introduction
'[STCNet: Alternating CNN and Improved Transformer Network for COVID-19 CT Image Segmentation](https://arxiv.org/abs/1811.0132)' submitted to ss on 12-March-2024

# STCNet

 Since the emergence of the Corona Virus Disease in 2019 (COVID-19), it has become a serious health problem affecting the human respiratory system. At present, automatic segmentation of lung infection areas from Computed Tomography has been playing a crucial role in the diagnosis of this disease because of its ability to perform pathological studies based on the lung infection areas. However, due to the lung infection areas scattered distribution, the existing segmentation methods generally have the problems of missing and incomplete segmentation. The Convolutional Neural Network (CNN)-based approaches generally lack the ability to model explicit long-range relation, and the transformer-based methods are not conducive to capturing the detailed boundaries of infected areas. Whereas the infected regions of the coronavirus images are scattered and boundary information plays an important role, both the boundaries and the global infected areas need to be taken into account. Therefore, we propose a novel coronavirus image segmentation network alternately using Swin transformer and CNN (STCNet). Firstly, to enable network to capture richer features, the ReSwin transformer block is proposed and added after each level of convolution block in the encoder-decoder. Secondly, to effectively retain the infected areas boundary information, the skip connection cross attention module is used to provide spatial information to each decoder. And through the fine-tuned scale-aware pyramid fusion module to fuse multi-scale context information. Experimental results show that STCNet at can achieve state-of-the-art performance on two coronavirus segmentation datasets, with Dice achieves 79.92% and 82.78%, respectively.


## Dataset

 COVID-19-CT-Seg dataset, CC-CCII Segmentation dataset

This repository is under development, please don't use this code
Questions
Please contact 'Gengpeng@stdu.edu.cn'
