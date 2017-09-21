"""
Evaluation metrics for graphs
Author: Diviyan Kalainathan
Date : 20/09
"""

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve


def count_true(tp, fp):
    try:
        tp.append(tp[-1] + 1)
        fp.append(fp[-1])

    except IndexError:
        tp.append(1)
        fp.append(0)
    return tp, fp


def count_false(tp, fp):
    try:
        tp.append(tp[-1])
        fp.append(fp[-1]+1)

    except IndexError:
        tp.append(0)
        fp.append(1)

    return tp, fp


def precision_recall(predictions, result):
    """ Sums up to Hamming distance on Directed Graphs

    :param predictions: list of Graphs or Graph
    :param result: DirectedGraph
    :return: list([aupr, precision, recall])
    """
    out = []
    if type(predictions) != list:
        predictions = [predictions]

    true_labels, true_nodes = result.adjacency_matrix()

    for pred in predictions:
        edges = pred.list_edges(descending=True, return_weights=True)
        m, nodes = pred.adjacency_matrix()
        predictions = pd.DataFrame(m, columns=nodes)
        if not set(true_nodes) == set(nodes):
            for i in (set(true_nodes) - set(nodes)):
                predictions[i] = 0
                predictions.loc[len(predictions)] = 0

        # Detect non-oriented edges and set a low value
        set_value_no = np.min(predictions.values[np.nonzero(predictions.values)])/2

        for e in edges:
            if [e[1], e[0], e[2]] in edges:
                predictions.loc[e[1], e[0]] = set_value_no
                predictions.loc[e[1], e[0]] = set_value_no

        predictions = predictions[true_nodes].as_matrix()
        precision, recall, _ = precision_recall_curve(true_labels.reshape(-1), predictions.reshape(-1))
        aupr = auc(recall, precision, reorder=True)
        out.append([aupr, precision, recall])

    return out