.. image:: banner.png

.. role:: raw-html(raw)
    :format: html

:raw-html:`<br />`


Causal Discovery Toolbox Documentation
======================================

Package for causal inference in graphs and in the pairwise settings for Python>=3.5. Tools for graph structure recovery and dependencies are included. The package is based on Numpy, Scikit-learn, Pytorch and R.

.. image:: https://travis-ci.org/FenTechSolutions/CausalDiscoveryToolbox.svg?branch=master
    :target: https://travis-ci.org/FenTechSolutions/CausalDiscoveryToolbox
.. image:: https://travis-ci.org/FenTechSolutions/CausalDiscoveryToolbox.svg?branch=dev
    :target: https://travis-ci.org/FenTechSolutions/CausalDiscoveryToolbox
.. image:: https://codecov.io/gh/FenTechSolutions/CausalDiscoveryToolbox/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/FenTechSolutions/CausalDiscoveryToolbox
.. image:: https://img.shields.io/aur/license/pac.svg?maxAge=259200
.. image:: https://img.shields.io/badge/version-0.5.21-yellow.svg?maxAge=259200

It implements lots of algorithms for graph structure recovery (including algorithms from the `bnlearn`, `pcalg` packages), mainly based out of observational data.

Install it using pip: (See more details on installation below)

.. code-block:: bash

   pip install cdt


Open-source project
===================

The package is open-source and under the MIT license, the source code is
available at : https://github.com/FenTechSolutions/CausalDiscoveryToolbox

When using this package, please cite: `Kalainathan, D., & Goudet, O. (2019).
Causal Discovery Toolbox: Uncover causal relationships in Python.
arXiv:1903.02278 <https://arxiv.org/abs/1903.02278>`_.

Docker images
=============

Docker images are available, including all the dependencies, and enabled functionalities:

.. |36cpu| image:: https://img.shields.io/badge/docker-0.5.21-0db7ed.svg?maxAge=259200
    :target: https://hub.docker.com/r/divkal/cdt-py3.6/

.. |37cpu| image:: https://img.shields.io/badge/docker-0.5.21-0db7ed.svg?maxAge=259200
    :target: https://hub.docker.com/r/divkal/cdt-py3.7/

.. |36gpu| image:: https://img.shields.io/badge/nvidia--docker-0.5.21-76b900.svg?maxAge=259200
    :target: https://hub.docker.com/r/divkal/nv-cdt-py3.6/

.. |36cpudev| image:: https://img.shields.io/badge/docker-latest-0db7ed.svg?maxAge=259200
    :target: https://hub.docker.com/r/divkal/cdt-dev/

.. |37cpudev| image:: https://img.shields.io/badge/docker-unavailable-lightgrey.svg?maxAge=259200

.. |36gpudev| image:: https://img.shields.io/badge/nvidia--docker-latest-76b900.svg?maxAge=259200
    :target: https://hub.docker.com/r/divkal/nv-cdt-dev/

..  |       Branch     |                                                                 master                                                                 |                                                                  dev                                                                 |
..  | Python 3.6 - CPU |       [![d36cpu](https://img.shields.io/badge/docker-0.5.21-0db7ed.svg?maxAge=259200)](https://hub.docker.com/r/divkal/cdt-py3.6/)      |       [![d36cpudev](https://img.shields.io/badge/dev-0.5.21-0db7ed.svg?maxAge=259200)](https://hub.docker.com/r/divkal/cdt-dev)       |
..  | Python 3.7 - CPU |       [![d37cpu](https://img.shields.io/badge/docker-0.5.21-0db7ed.svg?maxAge=259200)](https://hub.docker.com/r/divkal/cdt-py3.7/)      |                       [![d37gpu](https://img.shields.io/badge/dev-unavailable-lightgrey.svg?maxAge=259200)](#)                       |
..  | Python 3.6 - GPU | [![d36gpu](https://img.shields.io/badge/nvidia--docker-0.5.21-76b900.svg?maxAge=259200)](https://hub.docker.com/r/divkal/nv-cdt-py3.6/) |  [![d36gpudev](https://img.shields.io/badge/nvidia--dev-0.5.21-0db7ed.svg?maxAge=259200)](https://hub.docker.com/r/divkal/nv-cdt-dev) |

+-------------------+---------+------------+
|      Branch       | master  |    dev     |
+===================+=========+============+
|  Python 3.6 - CPU | |36cpu| | |36cpudev| |
+-------------------+---------+------------+
|  Python 3.7 - CPU | |37cpu| | |37cpudev| |
+-------------------+---------+------------+
|  Python 3.6 - GPU | |36gpu| | |36gpudev| |
+-------------------+---------+------------+


Installation
============

The packages requires a python version >=3.5, as well as some libraries listed
in the `requirements file <https://github.com/FenTechSolutions/CausalDiscoveryToolbox/blob/master/requirements.txt>`_.
For some additional functionalities, more libraries are needed for these extra
functions and options to become available. Here is a quick install guide of the
package, starting off with the minimal install up to the full installation.

.. note::
   A (mini/ana)conda framework would help installing all those packages and therefore could be recommended for non-expert users.

PyTorch
-------
As some of the key algorithms in the _cdt_ package use the PyTorch package, it is required to install it. 
Check out their website to install the PyTorch version suited to your hardware configuration: https://pytorch.org

Install the CausalDiscoveryToolbox package
------------------------------------------
The package is available on PyPi:

.. code-block:: bash

   pip install cdt

Or you can also install it from source.

.. code-block:: bash

   $ git clone https://github.com/FenTechSolutions/CausalDiscoveryToolbox.git  # Download the package
   $ cd CausalDiscoveryToolbox
   $ pip install -r requirements.txt  # Install the requirements
   $ python setup.py install develop --user

**The package is then up and running ! You can run most of the algorithms in the CausalDiscoveryToolbox, you might get warnings: some additional features are not available**

From now on, you can import the library using :

.. code-block:: python

   import cdt


Additional : R and R libraries
------------------------------
In order to have access to additional algorithms from various R packages such as bnlearn, kpcalg, pcalg, ... while using the _cdt_ framework, it is required to install R.

Check out how to install all R dependencies in the before-install section of the [travis.yml](https://github.com/FenTechSolutions/CausalDiscoveryToolbox/blob/master/.travis.yml) file for debian based distributions.
The `r-requirements file <https://github.com/FenTechSolutions/CausalDiscoveryToolbox/blob/master/r_requirements.txt>`_ notes all the R packages used by the toolbox.


Overview
========
The following figure shows how the package and its algorithms are structured:


::

   cdt package
   |
   |- independence
   |  |- graph (Infering the skeleton from data)
   |  |  |- Lasso variants (Randomized Lasso[1], Glasso[2], HSICLasso[3])
   |  |  |- FSGNN (CGNN[12] variant for feature selection)
   |  |  |- Skeleton recovery using feature selection algorithms (RFECV[5], LinearSVR[6], RRelief[7], ARD[8,9], DecisionTree)
   |  |
   |  |- stats (pairwise methods for dependency)
   |     |- Correlation (Pearson, Spearman, KendallTau)
   |     |- Kernel based (NormalizedHSIC[10])
   |     |- Mutual information based (MIRegression, Adjusted Mutual Information[11], Normalized mutual information[11])
   |
   |- data
   |  |- CausalPairGenerator (Generate causal pairs)
   |  |- AcyclicGraphGenerator (Generate FCM-based graphs)
   |  |- load_dataset (load standard benchmark datasets)
   |
   |- causality
   |  |- graph (methods for graph inference)
   |  |  |- CGNN[12]
   |  |  |- PC[13]
   |  |  |- GES[13]
   |  |  |- GIES[13]
   |  |  |- LiNGAM[13]
   |  |  |- CAM[13]
   |  |  |- GS[23]
   |  |  |- IAMB[24]
   |  |  |- MMPC[25]
   |  |  |- SAM[26]
   |  |  |- CCDr[27]
   |  |
   |  |- pairwise (methods for pairwise inference)
   |     |- ANM[14] (Additive Noise Model)
   |     |- IGCI[15] (Information Geometric Causal Inference)
   |     |- RCC[16] (Randomized Causation Coefficient)
   |     |- NCC[17] (Neural Causation Coefficient)
   |     |- GNN[12] (Generative Neural Network -- Part of CGNN )
   |     |- Bivariate fit (Baseline method of regression)
   |     |- Jarfo[20]
   |     |- CDS[20]
   |     |- RECI[28]
   |
   |- metrics (Implements the metrics for graph scoring)
   |  |- Precision Recall
   |  |- SHD
   |  |- SID [29]
   |
   |- utils
      |- Settings -> SETTINGS class (hardware settings)
      |- loss -> MMD loss [21, 22] & various other loss functions
      |- io -> for importing data formats
      |- graph -> graph utilities





References
==========

- [1] Wang, S., Nan, B., Rosset, S., & Zhu, J. (2011). Random lasso. The annals of applied statistics, 5(1), 468.

- [2] Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. Biostatistics, 9(3), 432-441.

- [3] Yamada, M., Jitkrittum, W., Sigal, L., Xing, E. P., & Sugiyama, M. (2014). High-dimensional feature selection by feature-wise kernelized lasso. Neural computation, 26(1), 185-207.

- [4] Feizi, S., Marbach, D., Médard, M., & Kellis, M. (2013). Network deconvolution as a general method to distinguish direct dependencies in networks. Nature biotechnology, 31(8), 726-733.

- [5] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. Machine learning, 46(1), 389-422.

- [6] Vapnik, V., Golowich, S. E., & Smola, A. J. (1997). Support vector method for function approximation, regression estimation and signal processing. In Advances in neural information processing systems (pp. 281-287).

- [7] Kira, K., & Rendell, L. A. (1992, July). The feature selection problem: Traditional methods and a new algorithm. In Aaai (Vol. 2, pp. 129-134).

- [8] MacKay,  D.  J.  (1992). Bayesian interpolation. Neural Computation, 4, 415–447.

- [9] Neal, R. M. (1996). Bayesian learning for neural networks. No. 118 in Lecture Notes in Statistics. New York: Springer.

- [10] Gretton, A., Bousquet, O., Smola, A., & Scholkopf, B. (2005, October). Measuring statistical dependence with Hilbert-Schmidt norms. In ALT (Vol. 16, pp. 63-78).

- [11] Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance. Journal of Machine Learning Research, 11(Oct), 2837-2854.

- [12] Goudet, O., Kalainathan, D., Caillou, P., Lopez-Paz, D., Guyon, I., Sebag, M., ... & Tubaro, P. (2017). Learning functional causal models with generative neural networks. arXiv preprint arXiv:1709.05321.

- [13] Spirtes, P., Glymour, C., Scheines, R. (2000). Causation, Prediction, and Search. MIT press.

- [14] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2009). Nonlinear causal discovery with additive noise models. In Advances in neural information processing systems (pp. 689-696).

- [15] Janzing, D., Mooij, J., Zhang, K., Lemeire, J., Zscheischler, J., Daniušis, P., ... & Schölkopf, B. (2012). Information-geometric approach to inferring causal directions. Artificial Intelligence, 182, 1-31.

- [16] Lopez-Paz, D., Muandet, K., Schölkopf, B., & Tolstikhin, I. (2015, June). Towards a learning theory of cause-effect inference. In International Conference on Machine Learning (pp. 1452-1461).

- [17] Lopez-Paz, D., Nishihara, R., Chintala, S., Schölkopf, B., & Bottou, L. (2017, July). Discovering causal signals in images. In Proceedings of CVPR.

- [18] Stegle, O., Janzing, D., Zhang, K., Mooij, J. M., & Schölkopf, B. (2010). Probabilistic latent variable models for distinguishing between cause and effect. In Advances in Neural Information Processing Systems (pp. 1687-1695).

- [19] Zhang, K., & Hyvärinen, A. (2009, June). On the identifiability of the post-nonlinear causal model. In Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence (pp. 647-655). AUAI Press.

- [20] Fonollosa, J. A. (2016). Conditional distribution variability measures for causality detection. arXiv preprint arXiv:1601.06680.

- [21] Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. Journal of Machine Learning Research, 13(Mar), 723-773.

- [22] Li, Y., Swersky, K., & Zemel, R. (2015). Generative moment matching networks. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15) (pp. 1718-1727).  

- [23] Margaritis D (2003). Learning Bayesian Network Model Structure from Data . Ph.D. thesis, School of Computer Science, Carnegie-Mellon University, Pittsburgh, PA. Available as Technical Report CMU-CS-03-153

- [24] Tsamardinos I, Aliferis CF, Statnikov A (2003). “Algorithms for Large Scale Markov Blanket Discovery”. In “Proceedings of the Sixteenth International Florida Artificial Intelligence Research Society Conference”, pp. 376-381. AAAI Press.

- [25] Tsamardinos I, Aliferis CF, Statnikov A (2003). “Time and Sample Efficient Discovery of Markov Blankets and Direct Causal Relations”. In “KDD ’03: Proceedings of the Ninth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining”, pp. 673-678. ACM. Tsamardinos I, Brown LE, Aliferis CF (2006). “The Max-Min Hill-Climbing Bayesian Network Structure Learning Algorithm”. Machine Learning,65(1), 31-78.

- [26] Kalainathan, Diviyan & Goudet, Olivier & Guyon, Isabelle & Lopez-Paz, David & Sebag, Michèle. (2018). SAM: Structural Agnostic Model, Causal Discovery and Penalized Adversarial Learning.

- [27] Aragam, B., & Zhou, Q. (2015). Concave penalized estimation of sparse Gaussian Bayesian networks. Journal of Machine Learning Research, 16, 2273-2328.

- [28] Bloebaum, P., Janzing, D., Washio, T., Shimizu, S., & Schoelkopf, B. (2018, March). Cause-Effect Inference by Comparing Regression Errors. In International Conference on Artificial Intelligence and Statistics (pp. 900-909).

- [29] Structural Intervention Distance (SID) for Evaluating Causal Graphs, Jonas Peters, Peter Bühlmann: https://arxiv.org/abs/1306.1043


.. Table of Contents

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 1 
   :hidden:

   tutorial
   causality
   independence
   data
   utils
   metrics
   settings
   models
   developer

Indices and tables
==================

:ref:`genindex`

:ref:`modindex`

:ref:`search`
