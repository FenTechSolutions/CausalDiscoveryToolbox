"""
Evaluation metrics for graphs
Author: Diviyan Kalainathan
Date : 20/09
"""

import numpy as np
from sklearn.metrics import auc

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

    for pred in predictions:
        edges = pred.list_edges(descending=True, return_weights=True)
        true_edges = result.list_edges()
        p = len(true_edges)  # number of positives

        noedges = []
        # Detect non-oriented edges
        for e in edges:
            if [e[1], e[0], e[2]] in edges:
                e.append(noedges)
                edges.remove(e)
                edges.remove([e[1], e[0], e[2]])

        tp = []
        fp = []
        previous_score = -1  # To detect if multiple edges have the same score
        count_multiple = 1  # No of edges that have the same score

        for e in edges:
            if previous_score == e[3]:
                count_multiple += 1
            else:
                count_multiple = 1

            if [e[0], e[1]] in true_edges:
                tp, fp = count_true(tp, fp)

            else:
                tp, fp = count_false(tp, fp)

            previous_score = e[3]
            if count_multiple > 1:  # suboptimal here
                tp[-count_multiple:] = [tp[-1] for i in range(count_multiple)]
                fp[-count_multiple:] = [fp[-1] for i in range(count_multiple)]

        tpnoe, fpnoe = tp, fp
        for noe in noedges:
            if [noe[0], noe[1]] in true_edges or [noe[0], noe[1]] in true_edges:
                tpnoe, fpnoe = count_true(tpnoe, fpnoe)
                tpnoe, fpnoe = count_false(tpnoe, fpnoe)

            else:
                tpnoe, fpnoe = count_false(tpnoe, fpnoe)
                tpnoe, fpnoe = count_false(tpnoe, fpnoe)

        tp.extend([tpnoe[-1] for i in range(len(noedges)*2)])
        fp.extend([fpnoe[-1] for i in range(len(noedges)*2)])

        precision = [i/p for i in tp]
        recall = [i/(i+j) for i, j in zip(tp, fp)]
        aupr = auc(recall, precision)
        out.append([aupr, precision, recall])

    return out