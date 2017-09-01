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
import cdt.utils.R

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid the thousands of stderr output lines

__all__ = ['DirectedGraph', 'UndirectedGraph']

del cdt.cdt
