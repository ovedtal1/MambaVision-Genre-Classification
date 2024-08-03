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

[YouTube](https://youtu.be/i8Cnas7QrMc) - https://youtu.be/i8Cnas7QrMc


- [Table of contents](#Outline)
  * [Background](#background)
  * [Dataset](#Dataset)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Quick start](#Quick-start)
  * [Analysing and testing](#Analysing-and-testing)
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


## Quick start

- Clone the repo:
```console
git clone https://github.com/ovedtal1/MambaVision-Genre-Classification.git
```
- Download the free GTZAN dataset form Kaggle: <a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification">GTZAN</a>
- Place the data in the main folder of the repo
- Run the 'Build Dataset.ipynb' step by step for dataset creation
- Follow the 'train_models.ipynb' for training the different models (MambaVision classifer & CNN)


## Analysing and testing

- Analyze you training results with the 'data_analysis.ipynb' script
- Run the 'genre_predictor.py' scripy with you own music and classify it!

## References
* Ali Hatamizadeh, Jan Kautz [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/abs/2407.08083)
* Tri Dao, Albert Gu [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
* [Pytorch implementation and pre-trained wieght for MambaVision](https://github.com/NVlabs/MambaVision?tab=readme-ov-file) [NVIDIA Research]
* Yuval Hoffman, Roee Hadar [Music-Genre-Classification-using-Transformers project](https://github.com/YuvalHoffman/Music-Genre-Classification-using-Transformers)
