"""
Base class for dependence models
Author: Diviyan Kalainathan
"""

from cdt.utils.Graph import UndirectedGraph


class IndependenceModel(object):
    """
    Base class for independence and utilities to recover the undirected graph out of data.
    """
    def __init__(self):
        super(IndependenceModel, self).__init__()

    def predict(self, a, b):
        """

        :param a: First Variable
        :param b: Second Variable
        :return: Proba independence score. A score close to 0 -> independent
        """
        raise NotImplementedError

    def create_undirected_graph(self, data):
        """ Build a skeleton using a pairwise independence criterion

        :param data: Raw data table
        :type data: pandas.DataFrame
        :return: Undirected graph
        :rtype: UndirectedGraph
        """
        graph = UndirectedGraph()

        for idx_i, i in enumerate(data.columns):
            for idx_j, j in enumerate(data.columns[idx_i+1:]):
                score = self.predict(data[i].as_matrix(), data[j].as_matrix())
                if abs(score) > 0.001:
                    graph.add(i, j, score)

        return graph
