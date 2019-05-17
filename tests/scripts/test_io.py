#!/usr/bin/env python

import pandas as pd
from cdt.utils.io import (read_list_edges,
                          read_adjacency_matrix,
                          read_causal_pairs)
import networkx as nx
import os


def test_read_causal_pairs():
    w = read_causal_pairs('{}/../datasets/Example_pairwise_pairs.csv'.format(os.path.dirname(os.path.realpath(__file__))))
    assert type(w) == pd.DataFrame
    data = pd.read_csv('{}/../datasets/Example_pairwise_pairs.csv'.format(os.path.dirname(os.path.realpath(__file__))))
    w = read_causal_pairs(data)
    assert type(w) == pd.DataFrame


def test_read_adj_mat():
    w = read_adjacency_matrix('{}/../datasets/Example_target_adj.csv'.format(os.path.dirname(os.path.realpath(__file__))))
    assert type(w) == nx.DiGraph
    data = pd.read_csv('{}/../datasets/Example_target_adj.csv'.format(os.path.dirname(os.path.realpath(__file__))))
    w = read_adjacency_matrix(data, directed=False)
    assert type(w) == nx.Graph


def test_read_list_edges():
    w = read_list_edges('{}/../datasets/Example_graph_target.csv'.format(os.path.dirname(os.path.realpath(__file__))))
    assert type(w) == nx.DiGraph
    data = pd.read_csv('{}/../datasets/Example_graph_target.csv'.format(os.path.dirname(os.path.realpath(__file__))))
    w = read_list_edges(data, directed=False)
    assert type(w) == nx.Graph


if __name__ == '__main__':
    test_read_causal_pairs()
    test_read_adj_mat()
    test_read_list_edges()
