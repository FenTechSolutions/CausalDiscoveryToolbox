# Specifications of the Causal Discovery Toolbox

## Subpackages 

1. Independence
   1. Statistical tests
       * Pearson's correlation [_OK_]
       * Spearman's correlation [_OK_]
       * Kendall's tau [_OK_]
       * Mutual Information Regression [_OK_]
       * Normalized HSIC [_OK_] 
       * Adjusted Mutual Information [_OK_] 
       * Normalized Mutual Information [_OK_]
       
   2. Skeleton recovery
       * FSGNN **[_Validate_]**
       * Glasso [_OK_]
       * HSIC Lasso **[_Develop Class_]** **[_Validate_]**
       * minet (Mrnet, ARACNe, CLR) **[_Develop Wrapper_]** **[_Validate_]**
       * Network Deconvolution [_OK_]
       * Feature Selection Models **[_Validate_]**
       
2. Causality
   1. Pairwise inference
      
      * IGCI **[_Validate_]**
      * PNL **[_Develop_]** **[_Validate_]**
      * GPI **[_Develop_]** **[_Validate_]**
      * GNN **[_Validate tf & th_]** - Experiments launched
      * ANM [_OK_]
      * CDS [_OK_]
      * Bivariate Fit [_OK_]
      * RCC [_OK_]
      * NCC **[_Validate_]** 
  
   2. Cofounder detection **[_Develop_]**
   3. Graph inference:
      * CGNN **[_Validate tf & th_]** - Experiments launched
      * Other (Lingam?) **[_Develop_]** 
      * Bayesian Methods **[_Develop_]** (Interface w/ **bnlearn** ?)

3. Time-series **[_Develop_]** 
4. Utils :
   * Graph Class and operations [_OK_] 
   * Graph visualisation **[_Develop_]** **[_Validate_]**
   * Log info ? 
   * Import datasets **[_Develop_]** **[_Validate_]**
   * Dataset generator :
       * Random Graph [_OK_]
       * Graph from data [_OK_]
   * Loading pretrained models **[_Develop_]** **[_Validate_]**
   * Metrics **[_Develop_]** **[_Validate_]**
   * Losses **[_Validate_]**
   * Settings [_OK_]
	  
5. Auto-Causality **[_Develop_]** **[_Validate_]**

## Types and important notes

**We should be able to create custom methods using each of these steps as building blocks**

In order to do that, some types might have to be implemented:
* Dependency graphs
* Partially oriented
* Fully oriented (already done -see CGNN code)

**Dependencies have to be easily installed**

Including all R packages

**~~The Matlab software base might have to be ported to python~~** 
(Too much dependencies already introduced + Not having to pay )

The bnlearn package might be used using an R wrapper (rpy2) [ToDo]

**Testing hypothesis/ simplifying assumptions ?** 

How can we do it on a new dataset?
Modify dataset to get ou of the simplifying assumptions case
Datasets types for simplifying assumptions and types of variables? 

**Types of variables**

Continous, Categorical, Binary?

**Warn User of satisfiability of assumptions/ applicability of algorithms**

**Load Pretrained models** 
