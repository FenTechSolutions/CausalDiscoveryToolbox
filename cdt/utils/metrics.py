"""Evaluation metrics for graphs.

Author: Diviyan Kalainathan
Date : 20/09
"""

import numpy as np
import networkx as nx
from sklearn.metrics import auc, precision_recall_curve


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
    raise NotImplemented
    return 0
