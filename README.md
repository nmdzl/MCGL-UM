# Data Augmentation View on Graph Convolutional Network and the Proposal of Monte Carlo Graph Learning

This repository is the official implementation of [Data Augmentation View on Graph Convolutional Network and the Proposal of Monte Carlo Graph Learning](url_arxiv).

## Requirements

The project runs under Python 3.8.3 with several required packages.
To install requirements, run:

```setup
pip install -r requirements.txt
```

(Recommended) It is highly recommended to train the models on GPU. If your computing device is GPU supported (e.g., NVIDIA GPU), please install CUDA first. Please follow [the official website of NVIDIA](https://developer.nvidia.com/cuda-downloads).

(Recommended) It is highly recommended to have Anaconda (4.8.3 for this project) installed on your operating system for PyTorch installation. To install Anaconda 4.8.3, please follow the instructions on [its official website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

(Required) To install PyTorch, please follow the instructions on [its official website](https://pytorch.org/get-started/locally/). Select the best fit options and run the command to install PyTorch locally.

For example, if you select Stable (1.5) version, Windows operating system, Conda packages, Python language, CUDA version 10.2, run the following command to install:

```install PyTorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

## Models

In the paper, we trained three different models -- GCN, GCN* and MCGL-UM. To train GCN and GCN*, please run train_GCN.py and specify the baseline argument as 1 an 2 respectively. To train MCGL-UM, please run train_MC_base.py.

You can check ps.py to see the default values of all arguments. You can run both scripts without specifying any arguments (using the default values). Note that, always make sure that you specify the correct baseline, dataset, acc_file (saves the training result) to avoid expected error and overwriting.

### Train models with specifying hyper-parameters

To train the models with specific hyper-parameters (optional, the default are the best hyper-parameters of CORA dataset on MCGL-UM model), run these commands for example:

Train GCN model:
```train GCN
python train_GCN.py --baseline 1 --dataset cora --acc_file sample_acc_GCN1_cora.csv --hidden 32 --weight_decay 0.0005 --lr 0.005 -dropout 0.7 --seed 0
```

Train GCN* model:
```train GCN*
python train_GCN.py --baseline 2 --dataset citeseer --acc_file sample_acc_GCN2_citeseer.csv --hidden 64 --weight_decay 0.001 --lr 0.05 -dropout 0.4 --seed 1
```

Train MCGL-UM model:
```train MCGL-UM
python train_MC_base.py --dataset pubmed --acc_file sample_acc_MCGLUM_pubmed.csv --hidden 128 --weight_decay 0.0001 --lr 0.05 --dropout 0.5 --batch_size 200 --seed 2
```

### Train models without specifying hyper-parameters (recommanded)

Once you have decided the best hyper-parameters, it is recommanded to fill them in the corresponding dictionary in train_best.py, and run train_best.py (supporting all three models) to avoid the repetitive input of hyper-parameters. You can train all models by these commands for example:

Train GCN model:
```train GCN without params
python train_best.py --model GCN --dataset ms_academic --acc_file sample_acc_GCN1_ms_academic.csv --seed 3
```

Train GCN* model:
```train GCN* without params
python train_best.py --model GCN* --dataset cora --acc_file sample_acc_GCN2_cora.csv --seed 4
```

Train MCGL-UM model:
```train MCGL-UM without params
python train_best.py --model MCGL-UM --dataset citeseer --acc_file sample_acc_MCGLUM_citeseer.csv --seed 5
```

### Train with different depth

Also, you can train GCN* and MCGL-UM models with different depth. For GCN*, please specify the depth argument; for MCGL-UM, please specify both trdep nad tsdep arguments for the depth of training and inference respectively. Run these commands for example:

Train GCN* model with depth of 10:
```train GCN* depth=10
python train_best.py --model GCN* --dataset pubmed --acc_file sample_acc_GCN2_pubmed_depth=10.csv --depth 10 --seed 6
```

Train MCGL-UM model with trdep of 10 and tsdep of 10:
```train MCGL-UM trdep=10 tsdep=10
python train_best.py --model MCGL-UM --dataset ms_academic --acc_file sample_acc_MCGLUM_ms_academic_trdep=10_tsdep=10.csv --trdep 10 --tsdep 10 --seed 7
```

### Train with different reduced noise rate

Moreover, you can train all three models with different reduced noise rate (see paper for clear definition), by specifying the noise_rate argument. Note that, if the passed argument is equal or larger than the noise rate in original dataset, there is no change to the original graph structure. The default value of noise_rate argument is 1.0, which maintains the original graph structure. Run these commands for example:

Train GCN model with nosie rate reduced to 10%:
```train GCN nr=0.1
python train_best.py --model GCN --dataset cora --acc_file sample_acc_GCN1_cora_nr=0.1.csv --noise_rate 0.1 --seed 8
```

Train GCN* model with nosie rate reduced to 10%:
```train GCN* nr=0.1
python train_best.py --model GCN* --dataset citeseer --acc_file sample_acc_GCN2_citeseer_nr=0.1.csv --noise_rate 0.1 --seed 9
```

Train MCGL-UM model with nosie rate reduced to 10%:
```train MCGL-UM nr=0.1
python train_best.py --model MCGL-UM --dataset pubmed --acc_file sample_acc_MCGLUM_pubmed_nr=0.1.csv --noise_rate 0.1 --seed 10
```
