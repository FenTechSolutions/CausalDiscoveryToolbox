"""Utilities for graph not included in Networkx.

Author: Diviyan Kalainathan
"""
import networkx as nx
from copy import deepcopy


def dagify_min_edge(g):
    """Input a graph and output a DAG.

    Heuristic: Reverse the edge with the lowest score of the cycle if possible,
               else remove it.
    """
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.simple_cycles(g).next()
        scores = []
        edges = []
        for i, j in zip(cycle[:1], cycle[:1]):
            edges.append((i, j))
            scores.append(g[i][j]['weight'])

        i, j = edges[scores.index(min(scores))]
        gc = deepcopy(g)
        gc.remove_edge(i, j)
        gc.add_edge(j, i)

        if len(list(nx.simple_cycles(gc))) < len(list(nx.simple_cycles(g))):
            g.add_edge(j, i, weight=min(scores))
        g.remove_edge(i, j)
    return g
