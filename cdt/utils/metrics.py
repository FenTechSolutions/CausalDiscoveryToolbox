"""Evaluation metrics for graphs.

Author: Diviyan Kalainathan
Date : 20/09
"""

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve


def precision_recall(target, pred):
    """Compute (area under the PR curve, precision, recall), metric of evaluation for directed graphs.

    :param predictions: list of Graphs or Graph
    :param result: DirectedGraph
    :return: list([aupr, precision, recall])
    """
    true_labels, true_nodes = target.adjacency_matrix()
    m, nodes = pred.adjacency_matrix()

    # Detect non-oriented edges and set a low value
    set_value_no = np.min(
        m[np.nonzero(m)]) / 2

    for i in range(m.shape[1] - 1):
        for j in range(i, m.shape[1]):
            if m[i, j] != 0 and m[i, j] == m[j, i]:
                m[i, j] = set_value_no
                m[j, i] = set_value_no

    predictions = pd.DataFrame(m, columns=nodes)
    if not set(true_nodes) == set(nodes):
        for i in (set(true_nodes) - set(nodes)):
            predictions[i] = 0
            predictions.loc[len(predictions)] = 0

    predictions = predictions[true_nodes].as_matrix()
    reorder = [nodes.index(i) for i in sorted(nodes, key=true_nodes.index)]
    predictions = predictions[reorder]
    precision, recall, _ = precision_recall_curve(
        true_labels.reshape(-1), predictions.reshape(-1))
    aupr = auc(recall, precision, reorder=True)

    return aupr, precision, recall


def SHD(target, pred):
    """Compute the Structural Hamming Distance."""
    raise NotImplemented
    return 0


def SID(target, pred):
    u"""Compute the Strutural Intervention Distance.

    Ref:  Structural Intervention Distance (SID) for Evaluating Causal Graphs,
    Jonas Peters, Peter Bühlmann, https://arxiv.org/abs/1306.1043
    """
    raise NotImplemented
    return 0
