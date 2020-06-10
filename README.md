# Data Augmentation View on Graph Convolutional Network and the Proposal of Monte Carlo Graph Learning

This repository is the official implementation of [Data Augmentation View on Graph Convolutional Network and the Proposal of Monte Carlo Graph Learning](url_arxiv).

## Requirements

To install requirements:

```setup
pip intall -r requirements.txt
```

## Training

To train the models in the paper, run these command:

```train
MCGL-UM: python train_MC_base.py --acc_file [path/to/acc_file] --dataset [choose from "cora", "citeseer", "pubmed" and "ms_academic"] --hidden 32 --weight_decay 0.005 --lr 0.05 --dropout 0.5 --batch_size 50 --trdep 2 --tsdep 2 --seed 0
GCN: python train_GCN.py --baseline 1 --acc_file [path/to/acc_file] --dataset [choose from "cora", "citeseer", "pubmed" and "ms_academic"] --hidden 32 --weight_decay 0.005 --lr 0.05 --dropout 0.5 --seed 0
GCN*: python train_GCN.py --baseline 2 --acc_file [path/to/acc_file] --dataset [choose from "cora", "citeseer", "pubmed" and "ms_academic"] --hidden 32 --weight_decay 0.005 --lr 0.05 --dropout 0.5 --depth 2 --seed 0
```
