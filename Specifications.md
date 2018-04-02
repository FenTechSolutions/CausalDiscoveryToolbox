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
       * HSIC Lasso [_OK_]
       * minet (Mrnet, ARACNe, CLR) [_OK_]
       * Network Deconvolution [_OK_]
       * Feature Selection Models [_OK_]

2. Causality
   1. Pairwise inference

      * IGCI **[_Validate_]**
      * PNL **[_Develop_]** **[_Validate_]**
      * GPI **[_Develop_]** **[_Validate_]**
      * GNN **[_Validate_]**
      * ANM [_OK_]
      * CDS [_OK_]
      * Bivariate Fit [_OK_]
      * RCC [_OK_]
      * NCC **[_Validate_]**

   2. Cofounder detection **[_Develop_]**
   3. Graph inference:
      * CGNN **[_Develop & Validate_]**
      * CAM [_OK_]
      * SAM [_OK_]
      * PC [_OK_]
      * GES [_OK_]
      * GIES [_OK_]
      * LiNGAM [_OK_]
      * MMHC **[_Develop & Validate_]**
      * GS **[_Develop & Validate_]**
      * D2C **[_Develop & Validate_]**
      * CCD **[_Develop & Validate_]**

3. Time-series **[_Develop_]**
4. Utils :
   * Graph Dagify **[_Develop_]** **[_Validate_]**
   * Log info ?
   * Import datasets **[_Develop_]** **[_Validate_]**
   * Dataset generator :
       * Random Graph [_OK_]
   * Metrics [_OK_]
   * Losses **[_Validate_]**
   * Settings [_OK_]

5. Auto-Causality **[_Develop_]** **[_Validate_]**

## Types and important notes

**We should be able to create custom methods using each of these steps as building blocks**

In order to do that, some types might have to be implemented:
* Dependency graphs
* Partially oriented **[ToDo]**
* Fully oriented (already done -see CGNN code)

-> init in Dependency graphs : from partially/fully oriented graphs **[ToDo]**

**Dependencies have to be easily installed**

Including all R packages

**~~The Matlab software base might have to be ported to python~~**
(Too much dependencies already introduced + Not having to pay for Matlab software )

The bnlearn package might be used using an R wrapper (rpy2) **[ToDo]**

**Testing hypothesis/ simplifying assumptions ?**

How can we do it on a new dataset?
Modify dataset to get ou of the simplifying assumptions case
Datasets types for simplifying assumptions and types of variables?

**Types of variables**

Continous, Categorical, Binary? **[ToDo] : Now only numerical is supported**

**Warn User of satisfiability of assumptions/ applicability of algorithms [Long term objective]**

**Load Pretrained models**
