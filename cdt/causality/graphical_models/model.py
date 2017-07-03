"""
Pairwise causal models base class
Author: Diviyan Kalainathan
Date : 7/06/2017
"""
from ...utils.Graph import DirectedGraph, UndirectedGraph


class GraphModel(object):
    """ Base class for all pairwise causal inference models

    Usage for undirected/directed graphs and CEPC df format.
    """
    def __init__(self):
        """ Init. """
        super(GraphModel, self).__init__()

    def predict(self, df_data, graph=None):
        """ Orient an undirected graph using the pairwise method defined by the subclass
        Requirement : Name of the nodes in the graph correspond to name of the variables in df_data

        :param df_data:
        :param graph: UndirectedGraph or DirectedGraph or None
        :return: Directed graph w/ weights
        :rtype: DirectedGraph
        """
        if graph is None:
            return self.create_graph_from_data(df_data)
        elif type(graph) == DirectedGraph:
            return self.orient_directed_graph(df_data, graph)
        elif type(graph) == UndirectedGraph:
            return self.orient_undirected_graph(df_data, graph)
        else:
            print('Unknown Graph type')
            raise ValueError

    def orient_undirected_graph(self, data, umg):

        raise NotImplementedError

    def orient_directed_graph(self, data, dag):

        raise NotImplementedError

    def create_graph_from_data(self, data):

        raise NotImplementedError


