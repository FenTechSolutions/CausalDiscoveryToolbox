"""Test pairwise causal Discovery models."""

import os
import pandas as pd
import networkx as nx
from cdt import SETTINGS
from cdt.causality.graph import (CAM, GS, GIES, IAMB, CCDr, GES,
                                 Fast_IAMB, PC, LiNGAM, SAM, MMPC,
                                 Inter_IAMB, SAMv1, CCDr)
from cdt.independence.stats import AdjMI


data_graph = pd.read_csv('{}/../datasets/Example_graph_numdata.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :5]

SETTINGS.verbose = False


def test_graph():
    for method in [GS, GIES,  IAMB, Fast_IAMB, CAM, CCDr,
                   PC, LiNGAM, CCDr, GES, MMPC, Inter_IAMB]:
        print(method)
        m = method()
        output1 = m.predict(data_graph)
        assert isinstance(output1, nx.DiGraph)
    return 0


def test_directed():
    for method in [GS, GIES,  IAMB, Fast_IAMB,
                   PC, GES, MMPC, Inter_IAMB]:
        print(method)
        m = method()
        output1 = m.predict(data_graph)
        print(nx.adj_matrix(output1).todense())
        output2 = m.predict(data_graph, output1)
        assert isinstance(output2, nx.DiGraph)
    return 0


def test_undirected():
    for method in [GS, GIES,  IAMB, Fast_IAMB,
                   PC, GES, MMPC, Inter_IAMB]:
        print(method)
        s = AdjMI()
        un = s.predict_undirected_graph(data_graph)
        m = method()
        output1 = m.predict(data_graph, un)
        assert isinstance(output1, nx.DiGraph)
    return 0


def test_SAM():
    m = SAM(train_epochs=10, test_epochs=10, nh=10, dnh=10, nruns=1, njobs=1)
    assert isinstance(m.predict(data_graph), nx.DiGraph)
    return 0

def test_SAMv1():
    m = SAMv1(train_epochs=10, test_epochs=10, nh=10, dnh=10, nruns=1, njobs=1)
    assert isinstance(m.predict(data_graph), nx.DiGraph)
    return 0


if __name__ == "__main__":
    # test_SAM()
    test_directed()
    test_undirected()
    test_graph()
