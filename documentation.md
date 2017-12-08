# Documentation


## General package structure
The following figure shows how the package and its algorithms are structured
**The references will be soon added**

```
cdt package
|
|- independence
|  |- skeleton (Infering the skeleton from data, and removing spurious connections)
|     |- Lasso variants (Randomized Lasso, Glasso, HSICLasso)
|     |- FSGNN (CGNN variant for feature selection)
|     |- Network deconvolution 
|     |- Skeleton recovery using feature selection algorithms (RFECV, LinearSVR, RRelief, ARD, DecisionTree)
|  |- stats (pairwise methods for dependency)
|     |- Correlation (Pearson, Spearman, KendallTau)
|     |- Kernel based (NormalizedHSIC)
|     |- Mutual information based (MIRegression, Adjusted Mutual Information, Normalized mutual information)
|
|- generators
|
|
|- causality
|  |- graph (methods for graph inference)
|     |- CGNN method (In tensorflow, pytorch version needs revision)
|     |- PC (using a the rpy2 R wrapper, Needs validation)
|  |- pairwise (methods for pairwise inference)
|     |- ANM (Additive Noise Model)
|     |- IGCI (Information Geometric Causal Inference)
|     |- RCC (Randomized Causation Coefficient)
|     |- NCC (Neural Causation Coefficient)
|     |- GNN (Generative Neural Network -- Part of CGNN )
|     |- Bivariate fit (Baseline method of regression)
|     |- GPI, PNL, Jarfo to implement
|
|- utils
   |- Settings -> CGNN_SETTINGS, SETTINGS (hardware settings)
   |- Loss -> MMD loss & various other loss functions
   |- metrics -> Implements the metrics for graph scoring
   |- Formats -> for importing data formats
   |- Graph -> defines the DirectedGraph and UndirectedGraph class (see below)
  
```
## Hardware and algorithm settings

## The graph class




