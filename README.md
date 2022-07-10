# WL-distance
This repository contains code which accompanies the paper [Gromov-Wasserstein meets Weisfeiler-Lehman](https://arxiv.org/abs/2202.02495). 

## Setup
To run the code, you will need the following packages: numpy, POT, networkx, torch-geometric, grakel

Additionally, we use the KSVM implemented in the [WTK library](https://github.com/BorgwardtLab/WTK) and the Wasserstein Weisfeiler-Lehman (WWL) kernel implemented [here](https://github.com/BorgwardtLab/WWL). 

Computations for the Weisfeiler-Lehman distance are included in the utils/distances.py file. Currently, node labels based on degree and size of the graph are supported. The classification experiments with both nearest neighbor and SVMs are included in the experiments folder. 
