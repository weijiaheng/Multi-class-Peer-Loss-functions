# Peer Loss functions

This repository is the (Multi-Class & Deep Learning) Pytorch implementation of "[Peer Loss Functions: Learning from Noisy Labels without Knowing Noise Rates](https://arxiv.org/abs/1910.03231)" accepted by ICML2020. 


## Required Packages & Environment
**Supported OS:** Windows, Linux, Mac OS X; Python: 3.6/3.7; 

**Deep Learning Library:** PyTorch (GPU required)

**Required Packages:** Numpy, Pandas, random, sklearn, tqdm, csv, torch (Keras is required if you want to estimate the noise transition matrix).


## Utilities
This repository includes:
📋 Multi-class implementation of Peer Loss functions;

📋 Peer Loss functions in Deep Learning;

📋 Dynamical tunning strategies of Peer Loss functions to further improve the performance.

Details of running ($\alpha-$weighted) Peer Loss functions on MNIST, Fashion MNIST, CIFAR-10, CIFAR-100 with different noise setting are mentioned in the `README.md` file in each folder.

The workflow of $\alpha-$ weighted Peer Loss functions comes to:

![Figure1](peernet.png)

## Citation

If you use our code, please cite the following paper:

```
@inproceedings{liu2020peer,
  title={Peer loss functions: Learning from noisy labels without knowing noise rates},
  author={Liu, Yang and Guo, Hongyi},
  booktitle={International Conference on Machine Learning},
  pages={6226--6236},
  year={2020},
  organization={PMLR}
}
```

## Related Code
📋 Peer Loss functions and its experiments on UCI datasets is available at:
**https://github.com/gohsyi/PeerLoss**

## Thanks for watching!
