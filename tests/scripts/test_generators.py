"""Testing generators."""

import networkx as nx
from cdt.data import AcyclicGraphGenerator, CausalPairGenerator
from cdt.data.causal_mechanisms import normal_noise, uniform_noise, gmm_cause, gaussian_cause
import pandas as pd
mechanisms = ['linear', 'polynomial', 'sigmoid_add',
              'sigmoid_mix', 'gp_add', 'gp_mix']


def test_acyclic_generators():
    for mechanism in mechanisms:
        agg, data = AcyclicGraphGenerator(mechanism, npoints=200, nodes=10, parents_max=3).generate()
        assert type(agg) == nx.DiGraph
        assert nx.is_directed_acyclic_graph(agg)

def test_causal_pairs():
    for mechanism in mechanisms:
        data, labels = CausalPairGenerator(mechanism).generate(10, npoints=200)
        assert type(data) == pd.DataFrame

# def test_cyclic_generators():
#     for mechanism in mechanisms:
#         cgg, data = CyclicGraphGenerator(mechanism, points=200, nodes=10, parents_max=3).generate(nb_steps=5, averaging=2)
#         assert type(cgg) == nx.DiGraph
#         assert not nx.is_directed_acyclic_graph(cgg)


def test_causes():
    for cause in [gmm_cause, gaussian_cause]:
            agg, data = AcyclicGraphGenerator("linear", npoints=200, nodes=10, parents_max=3, initial_variable_generator=cause).generate()
            assert type(agg) == nx.DiGraph
            assert nx.is_directed_acyclic_graph(agg)


def test_noises():
        for noise in ['normal', 'uniform']:
            agg, data = AcyclicGraphGenerator("linear", npoints=200, nodes=10, parents_max=3, noise=noise).generate()
            assert type(agg) == nx.DiGraph
            assert nx.is_directed_acyclic_graph(agg)


if __name__ == "__main__":
    test_causes()
    test_noises()
