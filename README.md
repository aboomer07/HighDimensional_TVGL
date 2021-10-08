# HighDimensional_TVGL

This project was done in collaboration with Jacob Pichelmann during my masters program at the Toulouse School of Economics.

## Overview
This presentation was the final project for my High Dimensional Models class at the Toulouse School of Economics. It is an introduction to gaussian graphical models, inducing network sparsity via regularization, and the alternating direction method of multipliers (ADMM) optimization method. It continues with a replication and senstitivty analysis extension of the paper "Network Inference via the Time-Varying Graphical Lasso" by David Hallac, which can be found [here](https://cs.stanford.edu/people/jure/pubs/tvgl-kdd17.pdf). The paper adds a regularization function to penalize change in the network structure over time, which allows for identifying various transitions and breaks in network structure. We extended their analysis by looking at an extended time period and an expanded set of stocks to test the robustness of their method.

## Data Sources
We replicated the authors application to a panel of stock price data.

## Tools Used
The slides were built using the beamer type in Latex. The project was coded in python, and used the Pandas, Numpy, Networkx, Sklearn, and Seaborn libraries.
