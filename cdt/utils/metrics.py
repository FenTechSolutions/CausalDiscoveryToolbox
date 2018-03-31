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


def precision_recall(target, pred):
    """Compute (area under the PR curve, precision, recall), metric of evaluation for directed graphs.

    :param predictions: Graph predicted, nx.DiGraph
    :param target: Target, nx.DiGraph
    :return: (aupr, precision, recall)
    """
    true_labels = np.array(nx.adjacency_matrix(target, weight=None).todense())
    predictions = np.array(nx.adjacency_matrix(pred, target.nodes()).todense())
    precision, recall, _ = precision_recall_curve(
        true_labels.ravel(), predictions.ravel())
    aupr = auc(recall, precision, reorder=True)

    return aupr, precision, recall


def SHD(target, pred, double_for_anticausal=True):
    """Compute the Structural Hamming Distance."""
    true_labels = np.array(nx.adjacency_matrix(target, weight=None).todense())
    predictions = np.array(nx.adjacency_matrix(pred, target.nodes(), weight=None).todense())

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

    if type(target) == nx.DiGraph:
        true_labels = np.array(nx.adjacency_matrix(target, weight=None).todense())
        predictions = np.array(nx.adjacency_matrix(pred, target.nodes(), weight=None).todense())
    elif type(target) == np.ndarray:
        true_labels = target
        predictions = pred
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
