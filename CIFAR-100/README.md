# Experiments on CIFAR-100

## Required Packages & Environment
**Supported OS:** Windows, Linux, Mac OS X; Python: 3.6/3.7; 
**Deep Learning Library:** PyTorch (GPU required)
**Required Packages:** Numpy, Pandas, random, matplotlib, tqdm, csv, torch.


## Training -- CIFAR-100
We give 4 noise CIFAR-100 dataset. To evaluate the performance of Peer Loss functions on different noise dataset, run:
Train CE model as a warm-up:
```
python3 runner.py --r noise --s seed --batchsize 128
```
Sparse-Low | Sparse-High | Uniform-Low | Uniform-High 
--- | --- | --- | --- 
r=0.1 | r=0.2 | r=0.3 | r=0.4

