"""Test stat methods."""


import os
import pandas as pd
import networkx as nx
from cdt.independence.stats import (AdjMI, NormMI, PearsonCorrelation,
                                    SpearmanCorrelation, KendallTau,
                                    NormalizedHSIC, MIRegression)


def init():
    return pd.read_csv('{}/../datasets/Example_graph_numdata.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :5]


def test_statistical_methods():
    data = init()
    for method in [AdjMI, NormMI, PearsonCorrelation, SpearmanCorrelation,
                   KendallTau, NormalizedHSIC, MIRegression]:
        model = method()
        assert type(model.predict_undirected_graph(data)) == nx.Graph
        assert model.predict(data.iloc[:, 0], data.iloc[:, 1]) != model.predict(data.iloc[:, 0], data.iloc[:, 0])
