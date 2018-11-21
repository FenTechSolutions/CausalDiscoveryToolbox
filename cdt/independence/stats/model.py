"""Base class for dependence models.

Author: Diviyan Kalainathan
"""
from networkx import Graph


class IndependenceModel(object):
    """Base class for independence and utilities to recover the
    undirected graph out of data.

    Args:
        predictor (function): function to estimate dependence (0 : independence), taking as input 2 array-like variables.

    """

    def __init__(self, predictor=None):
        """Init the model."""
        super(IndependenceModel, self).__init__()
        if predictor is not None:
            self.predict = predictor

    def predict(self, a, b):
        """Compute a dependence test statistic between variables.

        Args:
            a (numpy.ndarray): First variable
            b (numpy.ndarray): Second variable

        Returns:
            float: dependence test statistic (close to 0 -> independent)
        """
        raise NotImplementedError

    def predict_undirected_graph(self, data):
        """Build a skeleton using a pairwise independence criterion.

        Args:
            data (pandas.DataFrame): Raw data table

        Returns:
            networkx.Graph: Undirected graph representing the skeleton.
        """
        graph = Graph()

        for idx_i, i in enumerate(data.columns):
            for idx_j, j in enumerate(data.columns[idx_i+1:]):
                score = self.predict(data[i].values, data[j].values)
                if abs(score) > 0.001:
                    graph.add_edge(i, j, weight=score)

        return graph
