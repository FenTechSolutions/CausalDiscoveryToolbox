"""Test script for FSGNN, GNN and CGNN."""

# Import libraries
from cdt import SETTINGS
from cdt.independence.graph import FSGNN
from cdt.causality.graph import CGNN
# A warning on R libraries might occur. It is for the use of the r libraries that could be imported into the framework
import pandas as pd
import os
import networkx as nx

SETTINGS.NJOBS = 1


def test_pipeline_CGNN():
    data = pd.read_csv('{}/../datasets/NUM_LUCAS.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :3]

    # Finding the structure of the graph
    Fsgnn = FSGNN(train_epochs=5, test_epochs=5, l1=0.006)

    ugraph = Fsgnn.predict(data, threshold=1e-2)
    # List results
    # print(nx.adj_matrix(ugraph).todense().shape)
    # Orient the edges of the graph

    Cgnn = CGNN(nruns=1, train_epochs=2, test_epochs=2)
    assert type(Cgnn.predict(data, graph=ugraph)) == nx.DiGraph
    return 0


def test_multiprocessing_CGNN():
    data = pd.read_csv('{}/../datasets/NUM_LUCAS.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :3]

    # Finding the structure of the graph
    Fsgnn = FSGNN(train_epochs=5, test_epochs=5, l1=0.006, njobs=2)

    ugraph = Fsgnn.predict(data, threshold=1e-2)
    # List results
    # print(nx.adj_matrix(ugraph).todense().shape)
    # Orient the edges of the graph
    Cgnn = CGNN(nruns=6, njobs=2, train_epochs=2, test_epochs=2)
    assert type(Cgnn.predict(data, graph=ugraph)) == nx.DiGraph
    return 0


def test_multiprocessing_ete_CGNN():
    data = pd.read_csv('{}/../datasets/NUM_LUCAS.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :3]

    # Finding the structure of the graph

    # List results
    # print(nx.adj_matrix(ugraph).todense().shape)
    # Orient the edges of the graph
    SETTINGS.GPU = 0
    Cgnn = CGNN(nruns=6, njobs=2, train_epochs=2, test_epochs=2)
    assert type(Cgnn.predict(data)) == nx.DiGraph
    return 0


def test_scratch_CGNN():
    data = pd.read_csv('{}/../datasets/NUM_LUCAS.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :3]

    # Finding the structure of the graph
    # List results
    # print(nx.adj_matrix(ugraph).todense().shape)
    # Orient the edges of the graph

    Cgnn = CGNN(nruns=1, train_epochs=2, test_epochs=2)
    assert type(Cgnn.predict(data)) == nx.DiGraph

    return 0


if __name__ == '__main__':
    # test_pipeline_CGNN()
    # test_scratch_CGNN()
    # test_multiprocessing_ete_CGNN()
    test_multiprocessing_CGNN()
