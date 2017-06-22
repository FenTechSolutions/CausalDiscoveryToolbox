"""
Pairwise causal models base class
Author: Diviyan Kalainathan
Date : 7/06/2017
"""
from sklearn.preprocessing import scale
from ...utils.Graph import DirectedGraph


class GraphModel(object):
    """ Base class for all pairwise causal inference models

    Usage for undirected/directed graphs and CEPC df format.
    """
    def __init__(self):
        """ Init. """
        super(GraphModel, self).__init__()

    def predict_graph(self, df_data, graph=None):
        """ Orient an undirected graph using the pairwise method defined by the subclass
        Requirement : Name of the nodes in the graph correspond to name of the variables in df_data

        :param x: UndirectedGraph or DirectedGraph or None
        :param df_data:
        :return: Directed graph w/ weights
        :rtype: DirectedGraph
        """

        pass

    def orient_undirected_graph(self):

        raise NotImplementedError

    def orient_directed_graph(self):

        raise NotImplementedError

    def create_graph_from_data(self):

        raise NotImplementedError


