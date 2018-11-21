"""Build undirected graph out of raw data.

Author: Diviyan Kalainathan
Date: 1/06/17

"""
import networkx as nx
from sklearn.covariance import GraphLasso
from .model import GraphSkeletonModel, FeatureSelectionModel
from .HSICLasso import hsiclasso
import numpy as np


class Glasso(GraphSkeletonModel):
    """Graphical Lasso to find an adjacency matrix

    .. note::
       Ref : Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
       covariance estimation with the graphical lasso. Biostatistics, 9(3),
       432-441.
    """

    def __init__(self):
        super(Glasso, self).__init__()

    def predict(self, data, alpha=0.01, max_iter=2000, **kwargs):
        """ Predict the graph skeleton.

        Args:
            data (pandas.DataFrame): observational data
            alpha (float): regularization parameter
            max_iter (int): maximum number of iterations

        Returns:
            networkx.Graph: Graph skeleton
        """
        edge_model = GraphLasso(alpha=alpha, max_iter=max_iter)
        edge_model.fit(data.values)

        return nx.relabel_nodes(nx.DiGraph(edge_model.get_precision()),
                                {idx: i for idx, i in enumerate(data.columns)})


class HSICLasso(FeatureSelectionModel):
    """Graphical Lasso with a kernel-based independence test."""
    def __init__(self):
        super(HSICLasso, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        """For one variable, predict its neighbouring nodes.

        Args:
            df_features (pandas.DataFrame):
            df_target (pandas.Series):
            idx (int): (optional) for printing purposes
            kwargs (dict): additional options for algorithms

        Returns:
            list: scores of each feature relatively to the target

        .. warning::
           Not implemented. Implemented by the algorithms.
        """

        y = np.transpose(df_target.values)
        X = np.transpose(df_features.values)

        path, beta, A, lam = hsiclasso(X, y)

        return beta
