# Specifications of the Causal Discovery Toolbox

## Subpackages 

1. Dependency criteria
2. Deconvolution methods
3. Cofounder detection
4. Pairwise inference
5. Time-series
6. Graph inference
   1. Graph methods:
      * CGNN
      * Other (Lingam?)
   2. Bayesian Methods : 
      * Constraint based
      * Score based
      * Hybrid
 
7. Utils :
   * Graph visualisation 
   * Log info ? 
   * Import datasets/ dataset generator
   * Loading pretrained models
   * Metrics
	  
## Types and important notes

**We should be able to create custom methods using each of these steps as building blocks**

In order to do that, some types might have to be implemented:
* Dependency graphs
* Partially oriented
* Fully oriented (already done -see CGNN code)

**Dependencies have to be easily installed**

Including all R packages

**The Matlab software base might have to be ported to python**

The bnlearn package might be used using an R wrapper (rpy2?)

**Testing hypothesis/ simplifying assumptions ?** 

How can we do it on a new dataset?
Modify dataset to get ou of the simplifying assumptions case
Datasets types for simplifying assumptions and types of variables? 

**Types of variables**

Continous, Categorical, Binary?

**Warn User of satisfiability of assumptions/ applicability of algorithms?**

**Load Pretrained models ?** 
