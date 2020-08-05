"""Test pairwise causal Discovery models."""

import os
import pandas as pd
import networkx as nx
from cdt.causality.pairwise import (ANM, IGCI, BivariateFit, CDS,
                                    NCC, RCC, RECI, GNN)
from cdt.independence.graph import Glasso
from cdt.utils.io import read_causal_pairs
from cdt import SETTINGS
from cdt.data import load_dataset

SETTINGS.NJOBS = 1


train_data = read_causal_pairs("{}/../datasets/Example_pairwise_pairs.csv".format(os.path.dirname(os.path.realpath(__file__)))).iloc[:, :50]

train_target = pd.read_csv("{}/../datasets/Example_pairwise_targets.csv".format(os.path.dirname(os.path.realpath(__file__)))).iloc[:, :50].set_index("SampleID")

data_pairwise = read_causal_pairs("{}/../datasets/Example_pairwise_pairs.csv".format(os.path.dirname(os.path.realpath(__file__)))).iloc[0, :50]

data_graph = pd.read_csv('{}/../datasets/Example_graph_numdata.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :5]

graph_skeleton = Glasso().predict(data_graph)
tueb, labels = load_dataset('tuebingen')
tueb, labels = tueb[:10], labels[:10]


def test_pairwise():
    for method in [ANM, IGCI, BivariateFit, CDS, RCC, NCC, RECI]:  # Jarfo
        print(method)
        m = method()
        if hasattr(m, "fit"):
            m.fit(train_data, train_target)
        r = m.predict(data_pairwise)
        assert r is not None
        print(r)
    return 0


def test_pairwise():
    for method in [ANM, IGCI, BivariateFit, CDS, RCC, NCC, RECI]:  # Jarfo
        print(method)
        m = method()
        if hasattr(m, "fit"):
            m.fit(train_data, train_target)
        r = m.predict_dataset(tueb)
        assert r is not None
        print(r)
    return 0


def test_pairwise_GNN():
    method = GNN
    print(method)
    m = method(train_epochs=10, test_epochs=10, nruns=1)
    r = m.predict(data_pairwise)
    assert r is not None
    print(r)
    return 0


def test_graph_GNN():
    method = GNN
    print(method)
    m = method(train_epochs=10, test_epochs=10, nruns=2)
    assert type(m.predict(data_graph, graph_skeleton)) == nx.DiGraph
    return 0


def test_graph():
    for method in [ANM, IGCI, BivariateFit, CDS, RCC, RECI, NCC]:  # Jarfo
        print(method)
        m = method()
        if hasattr(m, "fit"):
            m.fit(train_data, train_target)
        assert type(m.predict(data_graph, graph_skeleton)) == nx.DiGraph
    return 0


def test_pairwise_t():
    for method in [NCC]:  # Jarfo
        print(method)
        m = method()
        if hasattr(m, "fit"):
            m.fit(tueb, labels)
        r = m.predict(data_pairwise)
        assert r is not None
        print(r)
    return 0

if __name__ == "__main__":
    test_pairwise()
    test_graph()
    test_pairwise_GNN()
    test_graph_GNN()
    test_pairwise_t()
