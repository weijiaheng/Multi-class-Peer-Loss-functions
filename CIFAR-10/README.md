# Experiments on CIFAR-10 


## Required Packages & Environment
**Supported OS:** Windows, Linux, Mac OS X; Python: 3.6/3.7; 
**Deep Learning Library:** PyTorch (GPU required)
**Required Packages:** Numpy, Pandas, random, matplotlib, tqdm, csv, torch.



## Training -- CIFAR-10
We give 6 noise CIFAR-10 dataset. To evaluate the performance of Peer Loss functions on different noise dataset, run: 
```
python3 runner.py --r noise --s seed --batchsize 128
```
Sparse-Low | Sparse-High | Uniform-Low | Uniform-High | Random-Low | Random-High 
--- | --- | --- | --- |--- |---
r=0.1 | r=0.2 | r=0.3 | r=0.4 | r=0.5 | r=0.6




