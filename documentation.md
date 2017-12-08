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
|  |- RandomGraphFromData (Generate a random graph similar to inputdata)
|  |- RandomGraphGenerator (Generates a random graph, can generate pairs of variables)
|  |- generate_graph_with_structure (generates a graph with a fixed structure)
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
The toolbox has a SETTINGS class as well as a CGNN_SETTINGS class that define respectively the hardware settings and the CGNN algorithm settings. Those settings are unique and their default parameters are defined in **_cdt/utils/Settings_**. 

These parameters are accessible and overridable via accessing the class : 

```python
import cdt
cdt.SETTINGS
cdt.CGNN_SETTINGS
```

Moreover, the hardware parameters are detected and defined automatically (including number of GPUs, CPUs, available optional packages) at the **import** of the package using the **cdt.utils.Settings.autoset_settings** method, run at startup. 

## The graph class
The whole package revolves around using the **DirectedGraph** and the **UndirectedGraph** classes that define how the graph has to be processed in the package. Those classes are accessible via :

```python
import cdt
cdt.DirectedGraph
cdt.UndirectedGraph
```
These classes are defined under the **cdt.utils.Graph.py** file. 



