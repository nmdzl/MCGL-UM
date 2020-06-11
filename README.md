# Data Augmentation View on Graph Convolutional Network and the Proposal of Monte Carlo Graph Learning

This repository is the official implementation of [Data Augmentation View on Graph Convolutional Network and the Proposal of Monte Carlo Graph Learning](url_arxiv).

## Requirements

The project runs under Python 3.8.3 with several required packages.
To install requirements, run:

```setup
pip install -r requirements.txt
```

(Optional) It is highly recommended to train the models on GPU. If your computing device is GPU supported (e.g., NVIDIA GPU), please install CUDA first. Please follow [the official website of NVIDIA](https://developer.nvidia.com/cuda-downloads).

(Optional) It is highly recommended to have Anaconda (4.8.3 for this project) installed on your operating system for PyTorch installation. To install Anaconda 4.8.3, please follow the instructions on [its official website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

(Required) To install PyTorch, please follow the instructions on [its official website](https://pytorch.org/get-started/locally/). Select the best fit options and run the command to install PyTorch locally.

For example, if you select Stable (1.5) version, Windows operating system, Conda packages, Python language, CUDA version 10.2, run the following command to install:
```install PyTorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

## Training

To train the models in the paper, run these command:

```train
MCGL-UM: python train_MC_base.py --acc_file [path/to/acc_file] --dataset [choose from "cora", "citeseer", "pubmed" and "ms_academic"] --hidden 32 --weight_decay 0.005 --lr 0.05 --dropout 0.5 --batch_size 50 --trdep 2 --tsdep 2 --seed 0
GCN: python train_GCN.py --baseline 1 --acc_file [path/to/acc_file] --dataset [choose from "cora", "citeseer", "pubmed" and "ms_academic"] --hidden 32 --weight_decay 0.005 --lr 0.05 --dropout 0.5 --seed 0
GCN*: python train_GCN.py --baseline 2 --acc_file [path/to/acc_file] --dataset [choose from "cora", "citeseer", "pubmed" and "ms_academic"] --hidden 32 --weight_decay 0.005 --lr 0.05 --dropout 0.5 --depth 2 --seed 0
```
