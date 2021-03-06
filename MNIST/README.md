# Experiments on MNIST 

## Required Packages & Environment
**Supported OS:** Windows, Linux, Mac OS X; Python: 3.6/3.7; 
**Deep Learning Library:** PyTorch (GPU required)
**Required Packages:** Numpy, Pandas, random, matplotlib, tqdm, csv, torch.



## Training -- MNIST
We give 6 noise MNIST dataset as mentioned in our paper. To run the experiment without bias correction:
```
python3 runner.py --r noise --batchsize 128
```


Corresponding noise file is:
Sparse-Low | Sparse-High | Uniform-Low | Uniform-High | Random-Low | Random-High
--- | --- | --- | --- | --- | ---
r=0.1 | r=0.2 | r=0.3 | r=0.4 | r=0.5 | r=0.6


