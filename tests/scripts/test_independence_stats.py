"""Test stat methods."""


import os
import pandas as pd
import numpy as np
import networkx as nx
from cdt.independence.stats import (AdjMI, NormMI, PearsonCorrelation,
                                    SpearmanCorrelation, KendallTau,
                                    NormalizedHSIC, MIRegression)

from cdt.utils.graph import dagify_min_edge


def init():
    return pd.read_csv('{}/../datasets/Example_graph_numdata.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :5]


def test_statistical_methods():
    data = init()
    for method in [AdjMI, NormMI, PearsonCorrelation, SpearmanCorrelation,
                   KendallTau, NormalizedHSIC, MIRegression]:
        model = method()
        assert type(model.predict_undirected_graph(data)) == nx.Graph
        assert model.predict(data.iloc[:, 0], data.iloc[:, 1]) != model.predict(data.iloc[:, 0], data.iloc[:, 0])


def test_dagify():
    graph = nx.DiGraph(np.random.uniform(size=(5,5)))
    for node in graph.nodes():
        graph.remove_edge(node, node)
    g2 = dagify_min_edge(graph)
    assert nx.is_directed_acyclic_graph(g2)


if __name__ == '__main__':
    test_dagify()
