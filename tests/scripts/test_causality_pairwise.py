"""Test pairwise causal Discovery models."""

import os
import pandas as pd
import networkx as nx
from cdt.causality.pairwise import (ANM, IGCI, Bivariate_fit, CDS, NCC, RCC)
from cdt.independence.graph import Glasso


train_data = pd.read_csv("{}/../datasets/Example_pairwise_pairs.csv".format(os.path.dirname(os.path.realpath(__file__)))).iloc[:, :50]

train_target = pd.read_csv("{}/../datasets/Example_pairwise_targets.csv".format(os.path.dirname(os.path.realpath(__file__)))).iloc[:, :50]

data_pairwise = pd.read_csv("{}/../datasets/Example_pairwise_pairs.csv".format(os.path.dirname(os.path.realpath(__file__)))).iloc[0, :50]

data_graph = pd.read_csv('{}/../datasets/Example_graph_numdata.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :5]

graph_skeleton = Glasso.predict(data_graph)


def test_pairwise():
    for method in [ANM, IGCI, Bivariate_fit, CDS, NCC, RCC]:
        m = method()
        if hasattr(m, "fit"):
            m.fit(train_data, train_target)
        m.predict(data_pairwise)


def test_graph():
    for method in [ANM, IGCI, Bivariate_fit, CDS, NCC, RCC]:
        m = method()
        if hasattr(m, "fit"):
            m.fit(train_data, train_target)
        assert type(m.predict(data_graph, graph_skeleton)) == nx.DiGraph
