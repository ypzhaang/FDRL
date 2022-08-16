# Federated Discriminative Representation Learning

Authors: Yupei Zhang~\IEEEmembership{IEEE Member}, Yifei Wang, Yuxin Li, Yunan Xu, Shuangshuang Wei, Shuhui Liu and Xuequn Shang

## Experiments

![Image text](https://github.com/ypzhaang/FDRL/blob/main/figure/fig4.pdf)

## Dependencies

The code requires Python >= 3.6 and PyTorch >= 1.2.0. 

## Data

This code uses the MNIST, Fashion-MNIST, CIFAR10 and CIFAR100 datasets.

The four datasets are downloaded automatically by the torchvision package. 

## Usage

FDRL is run using a command of the following form:

`python main_fdrl.py --alg fdrl --dataset [dataset] --num_users [num_users] --model [model] --shard_per_user [shard_per_user] --frac [frac] --lr [lr] --epochs [epochs] --local_ep [local_ep] --gpu [gpu]`

