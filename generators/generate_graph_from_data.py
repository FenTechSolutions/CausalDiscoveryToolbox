"""
Build random graphs based on unlabelled data
Author : Diviyan Kalainathan & Olivier Goudet
Date: 17/6/17
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from ..utils.Graph import DirectedGraph
from .generators import *
import matplotlib.pyplot as plt


class RandomGraphFromData(object):
    """ Generate a random graph out of data : produce a random graph and make statistics fit to the data

    """

    def __init__(self, df_data, simulator=polynomial_regressor,  datatype='Numerical'):
        """

        :param df_data:
        :param simulator:
        :param datatype:
        """
        super(RandomGraphFromData, self).__init__()
        self.data = df_data
        self.resimulated_data = None
        self.matrix_criterion = None
        self.llinks = None
        self.simulator = simulator
        try:
            assert datatype == 'Numerical'
            self.criterion = np.corrcoef
            self.matrix_criterion = True
        except AssertionError:
            print('Not Yet Implemented')
            raise NotImplementedError

    def find_dependencies(self, threshold=0.05):
        """ Find dependencies in the dataset out of the dataset

        :param threshold:
        """
        if self.matrix_criterion:
            corr = np.absolute(self.criterion(self.data.as_matrix()))
            np.fill_diagonal(corr, 0.)
        else:
            corr = np.zeros((len(self.data.columns), len(self.data.columns)))
            for idxi, i in enumerate(self.data.columns[:-1]):
                for idxj, j in enumerate(self.data.columns[idxi + 1:]):
                    corr[idxi, idxj] = np.absolute(
                        self.criterion(self.data[i], self.data[j]))
                    corr[idxj, idxi] = corr[idxi, idxj]

        self.llinks = [(self.data.columns[i], self.data.columns[j])
                       for i in range(len(self.data.columns) - 1)
                       for j in range(i + 1, len(self.data.columns)) if corr[i, j] > threshold]

    def generate_variables(self, graph, plot=False):
        generated_variables = {}
        nodes = graph.get_list_nodes()
        while len(generated_variables) < len(nodes):
            for var in nodes:
                par = graph.get_parents(var)
                if (var not in generated_variables and
                        set(par).issubset(generated_variables)):
                    # Variable can be generated
                    if len(par) == 0:
                        generated_variables[var] = self.data[var]
                    else:
                        generated_variables[var] = self.simulator(pd.DataFrame(generated_variables)[
                                                                      par].as_matrix(), self.data[var].as_matrix(),
                                                                  par).reshape(-1)
                        if plot:
                            if len(par) > 0:
                                plt.scatter(self.data[par[0]], self.data[var], alpha=0.2)
                                plt.scatter(generated_variables[par[0]], generated_variables[var], alpha=0.2)
                                plt.show()

        return generated_variables

    def generate_graph(self, draw_proba=.2):
        """ Generate random graph out of the data

        :param draw_proba:
        :return:
        """
        # Find dependencies
        if self.llinks is None:
            self.find_dependencies()

        # Draw random number of edges out of the dependent edges and create an
        # acyclic graph
        graph = DirectedGraph()

        for link in self.llinks:
            if np.random.uniform() < draw_proba:
                if np.random.uniform() < 0.5:
                    link = list(reversed(link))
                else:
                    link = list(link)

                # Test if adding the link does not create a cycle
                if not deepcopy(graph).add(link[0], link[1], 1).is_cyclic():
                    graph.add(link[0], link[1], 1)
                elif not deepcopy(graph).add(link[1], link[0], 1).is_cyclic():
                    # Test if we can add the link in the other direction
                    graph.add(link[1], link[0], 1)

        graph.remove_cycles()
        print(graph.is_cyclic(), graph.cycles())
        print('Adjacency matrix : {}'.format(graph.get_adjacency_matrix()))
        print('Number of edges : {}'.format(
            len(graph.get_list_edges(return_weights=False))))
        print("Beginning random graph build")
        print("Graph generated, passing to data generation!")
        # Resimulation of variables
        nodes = graph.get_list_nodes()

        # Regress using a y=P(Xc,E)= Sum_i,j^d(_alphaij*(X_1+..+X_c)^i*E^j) model & re-simulate data
        # run_graph_polynomial(self.data, graph,0,0)
        generated_variables = run_graph_polynomial(self.data, graph, 0, 0)

        return graph, pd.DataFrame(generated_variables, columns=nodes)
