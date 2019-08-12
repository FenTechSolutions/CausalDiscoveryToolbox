"""Acyclic Graph Generator.

Generates a dataset out of an acyclic FCM.
Author : Olivier Goudet and Diviyan Kalainathan

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
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
                                NN_Mechanism,
                                gmm_cause, normal_noise, uniform_noise)


class AcyclicGraphGenerator(object):
    """Generate an acyclic graph and data given a causal mechanism.

    Args:
        causal_mechanism (str): currently implemented mechanisms:
            ['linear', 'polynomial', 'sigmoid_add',
            'sigmoid_mix', 'gp_add', 'gp_mix', 'nn'].
        noise (str or function): type of noise to use in the generative process
            ('gaussian', 'uniform' or a custom noise function).
        noise_coeff (float): Proportion of noise in the mechanisms.
        initial_variable_generator (function): Function used to init variables
            of the graph, defaults to a Gaussian Mixture model.
        npoints (int): Number of data points to generate.
        nodes (int): Number of nodes in the graph to generate.
        parents_max (int): Maximum number of parents of a node.
        expected_degree (int): Degree (number of edge per node) expected,
            only used for erdos graph
        dag_type (str): type of graph to generate ('default', 'erdos')

    Example:
        >>> from cdt.data import AcyclicGraphGenerator
        >>> generator = AcyclicGraphGenerator('linear', npoints=1000)
        >>> data, graph = generator.generate()
        >>> generator.to_csv('generated_graph')
    """

    def __init__(self, causal_mechanism, noise='gaussian',
                 noise_coeff=.4,
                 initial_variable_generator=gmm_cause,
                 npoints=500, nodes=20, parents_max=5, expected_degree=3,
                 dag_type='default'):
        super(AcyclicGraphGenerator, self).__init__()
        self.mechanism = {'linear': LinearMechanism,
                          'polynomial': Polynomial_Mechanism,
                          'sigmoid_add': SigmoidAM_Mechanism,
                          'sigmoid_mix': SigmoidMix_Mechanism,
                          'gp_add': GaussianProcessAdd_Mechanism,
                          'gp_mix': GaussianProcessMix_Mechanism,
                          'nn': NN_Mechanism}[causal_mechanism]

        self.data = pd.DataFrame(None, columns=["V{}".format(i) for i in range(nodes)])
        self.nodes = nodes
        self.npoints = npoints
        try:
            self.noise = {'gaussian': normal_noise,
                          'uniform': uniform_noise}[noise]
        except KeyError:
            self.noise = noise
        self.noise_coeff = noise_coeff
        self.adjacency_matrix = np.zeros((nodes, nodes))
        self.parents_max = parents_max
        self.expected_degree = expected_degree
        self.dag_type = dag_type
        self.initial_generator = initial_variable_generator
        self.cfunctions = None
        self.g = None

    def init_dag(self, verbose):
        """Redefine the structure of the graph depending on dag_type
        ('default', 'erdos')

        Args:
            verbose (bool): Verbosity
        """
        if self.dag_type == 'default':
            for j in range(1, self.nodes):
                nb_parents = np.random.randint(0, min([self.parents_max, j])+1)
                for i in np.random.choice(range(0, j), nb_parents, replace=False):
                    self.adjacency_matrix[i, j] = 1

        elif self.dag_type == 'erdos':
            nb_edges = self.expected_degree * self.nodes
            prob_connection = 2 * nb_edges/(self.nodes**2 - self.nodes)
            causal_order = np.random.permutation(np.arange(self.nodes))

            for i in range(self.nodes - 1):
                node = causal_order[i]
                possible_parents = causal_order[(i+1):]
                num_parents = np.random.binomial(n=self.nodes - i - 1,
                                                 p=prob_connection)
                parents = np.random.choice(possible_parents, size=num_parents,
                                           replace=False)
                self.adjacency_matrix[parents, node] = 1

        try:
            self.g = nx.DiGraph(self.adjacency_matrix)
            assert not list(nx.simple_cycles(self.g))

        except AssertionError:
            if verbose:
                print("Regenerating, graph non valid...")
            self.init_dag(verbose=verbose)


    def init_variables(self, verbose=False):
        """Redefine the causes, mechanisms and the structure of the graph,
        called by ``self.generate()`` if never called.

        Args:
            verbose (bool): Verbosity
        """
        self.init_dag(verbose)

        # Mechanisms
        self.cfunctions = [self.mechanism(int(sum(self.adjacency_matrix[:, i])),
                                          self.npoints, self.noise, noise_coeff=self.noise_coeff)
                           if sum(self.adjacency_matrix[:, i])
                           else self.initial_generator for i in range(self.nodes)]

    def generate(self, rescale=True):
        """Generate data from an FCM defined in ``self.init_variables()``.

        Args:
            rescale (bool): rescale the generated data (recommended)

        Returns:
            tuple: (pandas.DataFrame, networkx.DiGraph), respectively the
            generated data and graph.
        """
        if self.cfunctions is None:
            self.init_variables()

        for i in nx.topological_sort(self.g):
            # Root cause

            if not sum(self.adjacency_matrix[:, i]):
                self.data['V{}'.format(i)] = self.cfunctions[i](self.npoints)
            # Generating causes
            else:
                self.data['V{}'.format(i)] = self.cfunctions[i](self.data.iloc[:, self.adjacency_matrix[:, i].nonzero()[0]].values)
            if rescale:
                self.data['V{}'.format(i)] = scale(self.data['V{}'.format(i)].values)

        return self.data, self.g

    def to_csv(self, fname_radical, **kwargs):
        """
        Save the generated data to the csv format by default,
        in two separate files: data, and the adjacency matrix of the
        corresponding graph.

        Args:
            fname_radical (str): radical of the file names. Completed by
               ``_data.csv`` for the data file and ``_target.csv`` for the
               adjacency matrix of the generated graph.
            \**kwargs: Optional keyword arguments can be passed to pandas.
        """
        if self.data is not None:
            self.data.to_csv(fname_radical+'_data.csv', index=False, **kwargs)
            pd.DataFrame(self.adjacency_matrix).to_csv(fname_radical \
                                                       + '_target.csv',
                                                       index=False, **kwargs)

        else:
            raise ValueError("Graph has not yet been generated. \
                              Use self.generate() to do so.")
