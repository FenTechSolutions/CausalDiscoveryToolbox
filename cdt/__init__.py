"""
The Causal Discovery Toolbox contains various methods for graph structure recovery and causal inference.
Additionally, it offers many utilities for graphs types.
It is CUDA-compatible for the most computationally expensive algorithms.
"""

import tensorflow as tf  # Required import or TensorFlow will crash with PyTorch
import os
import cdt.causality
import cdt.independence
import cdt.generators
from cdt.utils.Graph import DirectedGraph, UndirectedGraph
from cdt.utils import Loss
from cdt.utils.Settings import SETTINGS
from cdt.utils.R import RPackages
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid the thousands of stderr lines

try:
    import rpy2
    try:
        import readline
        import rpy2.robjects
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
        RPackages.minet = importr('minet')
        RPackages.pcalg = importr("pcalg")
        cdt.SETTINGS.r_is_available = True
    except rpy2.rinterface.RRuntimeError as e:
        cdt.SETTINGS.r_is_available = False
        warnings.warn("R wrapper is not available : {}".format(e))
except ImportError as e:
    cdt.SETTINGS.r_is_available = False
    warnings.warn("R wrapper is not available : {}".format(e))

__all__ = ['DirectedGraph', 'UndirectedGraph']

del cdt.cdt
