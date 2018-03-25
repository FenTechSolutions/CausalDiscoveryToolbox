"""Base class for dependence models.

Author: Diviyan Kalainathan
"""
from networkx import Graph


class IndependenceModel(object):
    """Base class for independence and utilities to recover the undirected graph out of data."""

    def __init__(self, predictor=None):
        """Init the model.

        :param predictor: function to estimate dependence (0 : independence),
                          taking as input 2 array-like variables.
        """
        super(IndependenceModel, self).__init__()
        if predictor is not None:
            self.predict = predictor

    def predict(self, a, b):
        """Test dependence between variables.

        :param a: First Variable
        :param b: Second Variable
        :return: Proba independence score. A score close to 0 -> independent
        """
        raise NotImplementedError

    def predict_undirected_graph(self, data):
        """Build a skeleton using a pairwise independence criterion.

        :param data: Raw data table
        :type data: pandas.DataFrame
        :return: Undirected graph
        :rtype: UndirectedGraph
        """
        graph = Graph()

        for idx_i, i in enumerate(data.columns):
            for idx_j, j in enumerate(data.columns[idx_i+1:]):
                score = self.predict(data[i].as_matrix(), data[j].as_matrix())
                if abs(score) > 0.001:
                    graph.add_edge(i, j, weight=score)

        return graph
