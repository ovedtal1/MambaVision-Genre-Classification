# MambaVision-Music-Genre-Classification

PyTorch implementation of music genre classification using MambaVision architecture

<h1 align="center">

  <img src="https://github.com/user-attachments/assets/cc32d342-57f0-49f8-86ff-b1c617621f4e" height="400">
</h1>
  <p align="center">
    <a href="https://github.com/ovedtal1">Tal Oved</a> •
    <a href="https://github.com/deanefraim1">Dean Efraim</a>
  </h1>
  <p align="center">
    <a href="https://il.linkedin.com/in/tal-oved-75b46b242">Linkedin</a> •
    <a href="https://il.linkedin.com/in/deanefraim?original_referer=https%3A%2F%2Fwww.google.com%2F">Linkedin</a>
  </p>

Video:

[YouTube](https://www.youtube.com/watch?v=A9I_awoJk64) 


- [Table of contents](#Table-of-contents)
  * [Background](#background)
  * [Dataset](#Dataset)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Quick start](#Quick-start)
  * [Analysing and testing](#Analysing-and-testing)
  * [Future Work](#Future-Work)
  * [References](#references)


## Background
The idea of our approach is to combine the sequential and time-dependent nature of music data with the MambaVision architecture for enhanced music genre classification. We leverage spectrograms as input, which are then processed by the MambaVision model, a lightweight transformer-like architecture tailored for feature extraction and patching. By doing so, we achieve superior results compared to traditional transformers and CNNs, even those pre-trained on different data types. The MambaVision model's ability to effectively handle the unique characteristics of musical spectrograms, coupled with its efficient feature extraction and patching capabilities, leads to significant improvements in classification performance. For detailed insights and theoretical underpinnings, please refer to our complete work.


## Dataset
The <a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification">GTZAN</a> dataset was used. The data set consists of 1000 songs in length of 30[sec] divided to 10 classes


## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.5.5`|
|`torch`|  `2.1.1`|
|`kornia`|  `0.7.3`|
|`matplotlib`|  `3.7.2`|
|`transformers`|  `4.42.3`|
|`numpy`|  `1.23.5`|
|`h5py`|  `3.10.0`|
|`librosa`|  `0.10.2`|
|`pandas`|  `2.1.1`|
|`seaborn`|  `0.13.0`|


## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`data_analysis.ipynb`| analysing the Model's results|
|`genre_predictor.py`| main script for spesific song prediction|
|`models.py`| contains all the models|
|`Paras.py`| initialize parameters for the project|
|`train_models.ipynb`| notebook for training the different models|
|`train.py`|  helper script for training the different models|
|`Build Dataset.ipynb`| notebook for step by step data prepearing|
|`data_loader.py`| data loading script|
|`music_dealer.py`| your own data loading script|
|`util.py`| utils for data use|


## Quick start

- Clone the repo:
```console
git clone https://github.com/ovedtal1/MambaVision-Genre-Classification.git
```
- Download the free GTZAN dataset form Kaggle: <a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification">GTZAN</a>
- Place the data in the main folder of the repo
- Run the 'Build Dataset.ipynb' step by step for dataset creation
- Follow the 'train_models.ipynb' for training the different models (MambaVision based, Transformer based & CNN)


## Analysing and testing

- Analyze your trained models with the 'data_analysis.ipynb' script
- Run the 'genre_predictor.ipynb' scripy with you own music and classify it!

## Future Work
- Compare the MambaVision with more architectures
- Search for more custom augmentations
- Test the MambaVision architecture on different tasks

## References
* Ali Hatamizadeh, Jan Kautz [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/abs/2407.08083)
* Lianghui Zhu et al. [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)
* Tri Dao, Albert Gu [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
* Mathilde Caron et al. [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
* [Pytorch implementation and pre-trained wieght for MambaVision](https://github.com/NVlabs/MambaVision?tab=readme-ov-file) [NVIDIA Research]
* [Self-Supervised Vision Transformers with DINO - pytorch](https://github.com/facebookresearch/dino?tab=readme-ov-file) [Facebook Research]
* Yuval Hoffman, Roee Hadar [Music-Genre-Classification-using-Transformers project](https://github.com/YuvalHoffman/Music-Genre-Classification-using-Transformers)
