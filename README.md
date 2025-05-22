# JacSim*: An Effective and Efficient Solution to the Pairwise Normalization Problem in SimRank

This repository provides the Python implementations of JasSim*, **both** Matrix form and Iterative form.

## Installation and usage
JacSim* is a recursive link-based similarity measure, which is applicable to **both** directed and undirected graphs. In the case of directed graphs, similarity scores can be computed based on _any_ of in-links or out-links. In order to use JacSim*, the following packages are required:
```
Python       >= 3.8
networkx     =2.6.*
numpy        =1.21.*
scikit-learn =1.0.*
```

**Graph file format:**

A1) A graph must be represented as a text file under the **edge list format** in which, each line corresponds to an edge in the graph, tab is used as the separator of the two nodes, and the node index is started from 0.

2) A single original link in an **undirected graph and its train/test splitted graphs** must be represented via two links in both directions.

## Citation:
> Masoud Reyhani Hamedani and Sang-Wook Kim. 2021. JacSim*: An Effective and Efficient Solution to the Pairwise Normalization Problem in SimRank. IEEE Access, Vol. 9, pages= 146038-146049. https://doi.org/10.1109/ACCESS.2021.3123114
