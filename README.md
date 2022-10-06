# Federated Discriminative Representation Learning

Authors: Yupei Zhang IEEE Member, Yifei Wang, Yuxin Li, Yunan Xu, Shuangshuang Wei, Shuhui Liu and Xuequn Shang

## Experiments

We use four public datasets: MNIST, Fashion-MNIST, CIFAR10 and CIFAR100

<img width="600px" src="https://github.com/ypzhaang/FDRL/blob/main/figure/dataset.jpg">

Here, we show 2D scatter plots with features：

<img width="600px" src="https://github.com/ypzhaang/FDRL/blob/main/figure/scatter.jpg">

The accuracy compared with other FL methods in MNIST：

|  Method   | Accuracy   |
|  ----  | ----  |
| FedAvg  | 96.34 |
| FedProx  | 97.20 |
| FedRep  | 97.72 |
| FDRL  | 98.41 |

## Proof

Let $\mathcal{G}_t$ be the local objective function of client $t$, a.k.a, the model in the client. Then FDRL aims to

$$ arg\min_{\mathbf{W}_t} \mathcal{G}_t=arg\min_{\mathbf{W}_t}  \sum_{i=1}^{n_t} \mathcal{L}_t(\mathcal{F}_t( \mathcal{M}_{\mathbf{\widetilde{W}}^t}(\mathbf{x}_i), \mathcal{M}_{\mathbf{\widehat{W}}^t}(\mathbf{x}_i))) + \alpha  \mathcal{S}_m(\mathcal{M}_{\mathbf{\widetilde{W}}^t},\mathcal{M}_{\mathbf{\widehat{W}}^t})
    + \beta  \mathcal{S}_r(\mathcal{M}_{\mathbf{\widetilde{W}}^t},\mathcal{M}_{\mathbf{\widehat{W}}^t}) $$
    
where $\mathcal{M}_{\mathbf{\widetilde{W}}^t}$ and $\mathcal{M}_{\mathbf{\widehat{W}}^t}$ are the shared model and the not-shared model in client $t$, $\mathcal{F}_t(\cdot)$ is the integration function, $\mathcal{S}_m(\cdot)$ is the model distance function and $\mathcal{S}_r(\cdot)$ is the representation distance function. Both $\mathcal{S}_m(\cdot)$ and $\mathcal{S}_r(\cdot)$ are convex functions.

TO BE CONVENIENT in proofs, we consider FDRL sharing the entire model but just returning the same parameters expect for the shared sub-model. Let $\mathcal{G}$ be the global objective function, that is

$$arg\min_{\mathbf{W^*}}\mathcal{G}=arg\min_{\mathbf{W^*}} \sum_{t=1}^m \mathcal{G}_{\mathbf{W^*}} (\mathcal{G}_t)$$

where $\mathbf{W^*}$ is the parameter of the global model.

## Dependencies

The code requires Python >= 3.6 and PyTorch >= 1.2.0. 

## Data

This code uses the MNIST, Fashion-MNIST, CIFAR10 and CIFAR100 datasets.

The four datasets are downloaded automatically by the torchvision package. 

## Usage

FDRL is run using a command of the following form:

`python main_fdrl.py --alg fdrl --dataset [dataset] --num_users [num_users] --model [model] --shard_per_user [shard_per_user] --frac [frac] --lr [lr] --epochs [epochs] --local_ep [local_ep] --gpu [gpu]`

