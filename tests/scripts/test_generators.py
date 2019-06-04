"""Testing generators."""

import networkx as nx
from cdt.data import AcyclicGraphGenerator, CausalPairGenerator
from cdt.data.causal_mechanisms import gmm_cause, gaussian_cause
import pandas as pd
import os


mechanisms = ['linear', 'polynomial', 'sigmoid_add',
              'sigmoid_mix', 'gp_add', 'gp_mix', 'nn']


def test_acyclic_generators():
    for mechanism in mechanisms:
        g = AcyclicGraphGenerator(mechanism, npoints=200, nodes=10, parents_max=3)
        data, agg = g.generate()
        g.to_csv('test')
        # cleanup
        os.remove('test_data.csv')
        os.remove('test_target.csv')
        assert type(agg) == nx.DiGraph
        assert nx.is_directed_acyclic_graph(agg)


def test_error():
    g = AcyclicGraphGenerator('linear', npoints=200, nodes=10, parents_max=3)
    try:
        g.to_csv('test')
    except ValueError:
        pass


def test_causal_pairs():
    for mechanism in mechanisms:
        data, labels = CausalPairGenerator(mechanism).generate(10, npoints=200)
        assert type(data) == pd.DataFrame


def test_acyclic_generators_bigg():
    for mechanism in mechanisms:
        data, agg = AcyclicGraphGenerator(mechanism, npoints=500, nodes=100, parents_max=6).generate()
        assert type(agg) == nx.DiGraph
        assert nx.is_directed_acyclic_graph(agg)

# def test_cyclic_generators():
#     for mechanism in mechanisms:
#         cgg, data = CyclicGraphGenerator(mechanism, points=200, nodes=10, parents_max=3).generate(nb_steps=5, averaging=2)
#         assert type(cgg) == nx.DiGraph
#         assert not nx.is_directed_acyclic_graph(cgg)


def test_causes():
    for cause in [gmm_cause, gaussian_cause]:
            data, agg = AcyclicGraphGenerator("linear", npoints=200, nodes=10, parents_max=3, initial_variable_generator=cause).generate()
            assert type(agg) == nx.DiGraph
            assert nx.is_directed_acyclic_graph(agg)


def test_noises():
        for noise in ['gaussian', 'uniform']:
            data, agg = AcyclicGraphGenerator("linear", npoints=200, nodes=10, parents_max=3, noise=noise).generate()
            assert type(agg) == nx.DiGraph
            assert nx.is_directed_acyclic_graph(agg)


if __name__ == "__main__":
    test_acyclic_generators()
    test_causal_pairs()
    test_causes()
    test_noises()
