"""Cyclic Graph Generator.

Generates a cross-sectional dataset out of a cyclic FCM.
Author : Olivier Goudet and Diviyan Kalainathan
"""

from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import networkx as nx
from .causal_mechanisms import (LinearMechanism,
                                Polynomial_Mechanism,
                                SigmoidAM_Mechanism,
                                SigmoidMix_Mechanism,
                                GaussianProcessAdd_Mechanism,
                                GaussianProcessMix_Mechanism,
                                normal_noise, gaussian_cause)


class CyclicGraphGenerator(object):
    """Generates a cross-sectional dataset out of a cyclic FCM."""

    def __init__(self, causal_mechanism, noise=normal_noise,
                 noise_coeff=.4,
                 initial_variable_generator=gaussian_cause,
                 points=500, nodes=20, timesteps=0, parents_max=5):
        """Generate an cyclic graph, given a causal mechanism.

        :param initial_variable_generator: init variables of the graph
        :param causal_mechanism: generating causes in the graph to
            choose between: ['linear', 'polynomial', 'sigmoid_add',
            'sigmoid_mix', 'gp_add', 'gp_mix']
        """
        super(CyclicGraphGenerator, self).__init__()
        self.mechanism = {'linear': LinearMechanism,
                          'polynomial': Polynomial_Mechanism,
                          'sigmoid_add': SigmoidAM_Mechanism,
                          'sigmoid_mix': SigmoidMix_Mechanism,
                          'gp_add': GaussianProcessAdd_Mechanism,
                          'gp_mix': GaussianProcessMix_Mechanism}[causal_mechanism]
        self.data = pd.DataFrame(None, columns=["V{}".format(i) for i in range(nodes)])
        self.nodes = nodes
        if timesteps == 0:
            self.timesteps = np.inf
        else:
            self.timesteps = timesteps
        self.points = points
        self.noise = noise
        self.noise_coeff = noise_coeff
        self.adjacency_matrix = np.zeros((nodes, nodes))
        self.parents_max = parents_max
        self.initial_generator = initial_variable_generator
        self.cfunctions = None
        self.g = None

    def init_variables(self, verbose=False):
        """Redefine the causes of the graph."""
        # Resetting adjacency matrix
        for i in range(self.nodes):
            for j in np.random.choice(range(self.nodes),
                                      np.random.randint(
                                          0, self.parents_max + 1),
                                      replace=False):
                if i != j:
                    self.adjacency_matrix[j, i] = 1

        try:
            assert any([sum(self.adjacency_matrix[:, i]) ==
                        self.parents_max for i in range(self.nodes)])
            self.g = nx.DiGraph(self.adjacency_matrix)
            assert list(nx.simple_cycles(self.g))
            assert any(len(i) == 2 for i in nx.simple_cycles(self.g))

        except AssertionError:
            if verbose:
                print("Regenerating, graph non valid...")
            self.init_variables()

        if verbose:
            print("Matrix generated ! \
              Number of cycles: {}".format(len(list(nx.simple_cycles(self.g)))))

        for i in range(self.nodes):
            self.data.iloc[:, i] = scale(self.initial_generator(self.points))

        # Mechanisms
        self.cfunctions = [self.mechanism(int(sum(self.adjacency_matrix[:, i])),
                                          self.points, self.noise, noise_coeff=self.noise_coeff) for i in range(self.nodes)]

    def generate(self, nb_steps=100, averaging=50, rescale=True):
        """Generate data from an FCM containing cycles."""
        if self.cfunctions is None:
            self.init_variables()
        new_df = pd.DataFrame()
        causes = [[c for c in np.nonzero(self.adjacency_matrix[:, j])[0]]
                  for j in range(self.nodes)]
        values = [[] for i in range(self.nodes)]

        for i in range(nb_steps):
            for j in range(self.nodes):
                new_df["V" + str(j)] = self.cfunctions[j](self.data.iloc[:, causes[j]].as_matrix())[:, 0]
                if rescale:
                    new_df["V" + str(j)] = scale(new_df["V" + str(j)])
                if i > nb_steps-averaging:
                    values[j].append(new_df["V" + str(j)])
            self.data = new_df
        self.data = pd.DataFrame(np.array([np.mean(values[i], axis=0)
                                           for i in range(self.nodes)]).transpose(),
                                 columns=["V{}".format(j) for j in range(self.nodes)])

        return self.g, self.data

    def to_csv(self, fname_radical, **kwargs):
        """
        Save data to the csv format by default, in two separate files.

        Optional keyword arguments can be passed to pandas.
        """
        if self.data is not None:
            self.data.to_csv(fname_radical+'_data.csv', **kwargs)
            pd.DataFrame(self.adjacency_matrix).to_csv(fname_radical+'_target.csv', **kwargs)

        else:
            raise ValueError("Graph has not yet been generated. \
                              Use self.generate() to do so.")


if __name__ == "__main__":
    print("Testing cyclic graph generator...")
    raise(NotImplemented)
