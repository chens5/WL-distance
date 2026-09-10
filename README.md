# WL-distance
This repository contains code which accompanies the paper [Gromov-Wasserstein meets Weisfeiler-Lehman](https://arxiv.org/abs/2202.02495). 

## Setup
To run the code, you will need the following packages: numpy, POT, networkx, torch-geometric, grakel, scikit-learn

The KSVM in `experiments/ksvm_utils.py` is a direct Python implementation of Algorithm 1 in [Learning SVM in Krein Spaces](https://doi.org/10.1109/TPAMI.2015.2477830). We also use the Wasserstein Weisfeiler-Lehman (WWL) kernel implemented [here](https://github.com/BorgwardtLab/WWL).

Computations for the Weisfeiler-Lehman distance are included in the utils/distances.py file. Currently, node labels based on degree and size of the graph are supported. The classification experiments with both nearest neighbor and SVMs are included in the experiments folder. 
