"""Evaluation metrics for graphs.

Author: Diviyan Kalainathan
Date : 20/09
"""

import os
import numpy as np
import networkx as nx
from shutil import rmtree
from sklearn.metrics import auc, precision_recall_curve
from .R import launch_R_script, RPackages


def precision_recall(target, prediction, low_confidence_undirected=False):
    r"""Compute precision-recall statistics for directed graphs.
    
    Precision recall statistics are useful to compare algorithms that make 
    predictions with a confidence score. Using these statistics, performance 
    of an algorithms given a set threshold (confidence score) can be approximated.
    Area under the precision-recall curve, as well as the coordinates of the 
    precision recall curve are computed, using the scikit-learn library tools.
    Note that unlike the AUROC metric, this metric does not account for class
    imbalance.

    Precision is defined by: :math:`Pr=tp/(tp+fp)` and directly denotes the
    total classification accuracy given a confidence threshold. On the other
    hand, Recall is defined by: :math:`Re=tp/(tp+fn)` and denotes  
    misclassification given a threshold.

    Args:
        target: Target graph, must be of ones and zeros, and instance of 
          either np.ndarray or nx.DiGraph.
        prediction: Prediction made by the algorithm to evaluate, must be 
          either np.ndarray or nx.DiGraph, but of the same type 
          than the target.
        low_confidence_undirected: Put the lowest confidence possible to 
          undirected edges (edges that are symmetric in the confidence score).
          Default: False
 
    Returns:
        aupr_score: the area under the precision recall curve
        precision_recall_points: tuple of data points precision-recall used 
                                 in for the area under the curve computation.

    Examples::
        >>> import numpy as np
        >>> tar, pred = np.random.randint(2, size=(10, 10)), np.random.randn(10, 10)
        >>> # adjacency matrixes of size 10x10
        >>> aupr, curve = precision_recall(target, input) 
        >>> # leave low_confidence_undirected to False as the predictions are continuous
    """
    if isinstance(target, np.ndarray):
        true_labels = target
        predictions = pred
    elif isinstance(target, nx.DiGraph):
        true_labels = np.array(nx.adjacency_matrix(target, weight=None).todense())
        predictions = np.array(nx.adjacency_matrix(pred, target.nodes()).todense())
    else:
        raise TypeError("Only networkx.DiGraph and np.ndarray (adjacency matrixes) are supported.")

    if low_confidence_undirected:
        # Take account of undirected edges by putting them with low confidence
        pred[pred==pred.transpose()] *= min(min(pred[np.nonzero(pred)])*.5, .1)
    precision, recall, _ = precision_recall_curve(
        true_labels.ravel(), pred.ravel())
    aupr = auc(recall, precision, reorder=True)

    return aupr, list(zip(precision, recall))


def SHD(target, pred, double_for_anticausal=True):
    """Compute the Structural Hamming Distance."""
    if isinstance(target, np.ndarray):
        true_labels = target
        predictions = pred
    elif isinstance(target, nx.DiGraph):
        true_labels = np.array(nx.adjacency_matrix(target, weight=None).todense())
        predictions = np.array(nx.adjacency_matrix(pred, target.nodes(), weight=None).todense())
    else:
        raise TypeError("Only networkx.DiGraph and np.ndarray (adjacency matrixes) are supported.")

    diff = np.abs(true_labels - predictions)
    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff)/2


def SID(target, pred):
    u"""Compute the Strutural Intervention Distance.

    Ref:  Structural Intervention Distance (SID) for Evaluating Causal Graphs,
    Jonas Peters, Peter Bühlmann, https://arxiv.org/abs/1306.1043
    """
    if not RPackages.SID:
        raise ImportError("SID R package is not available. Please check your installation.")

    if isinstance(target, np.ndarray):
        true_labels = target
        predictions = pred
    elif isinstance(target, nx.DiGraph):
        true_labels = np.array(nx.adjacency_matrix(target, weight=None).todense())
        predictions = np.array(nx.adjacency_matrix(pred, target.nodes(), weight=None).todense())
    else:
        raise TypeError("Only networkx.DiGraph and np.ndarray (adjacency matrixes) are supported.")

    os.makedirs('/tmp/cdt_SID/')

    def retrieve_result():
        return np.loadtxt('/tmp/cdt_SID/result.csv')

    try:
        np.savetxt('/tmp/cdt_SID/target.csv', true_labels, delimiter=',')
        np.savetxt('/tmp/cdt_SID/pred.csv', predictions, delimiter=',')
        sid_score = launch_R_script("{}/R_templates/sid.R".format(os.path.dirname(os.path.realpath(__file__))),
                                    {"{target}": '/tmp/cdt_SID/target.csv',
                                     "{prediction}": '/tmp/cdt_SID/pred.csv',
                                     "{result}": '/tmp/cdt_SID/result.csv'},
                                    output_function=retrieve_result)
    # Cleanup
    except Exception as e:
        rmtree('/tmp/cdt_SID')
        raise e
    except KeyboardInterrupt:
        rmtree('/tmp/cdt_SID/')
        raise KeyboardInterrupt

    rmtree('/tmp/cdt_SID')
    return sid_score
