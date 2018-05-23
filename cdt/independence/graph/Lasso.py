"""Build undirected graph out of raw data.

Author: Diviyan Kalainathan
Date: 1/06/17

"""
import networkx as nx
from sklearn.covariance import GraphLasso
from .model import GraphSkeletonModel, FeatureSelectionModel
from sklearn.linear_model import RandomizedLasso
from .HSICLasso import *


class Glasso(GraphSkeletonModel):
    """Apply Glasso to find an adjacency matrix

    Ref : ToDo - P.Buhlmann
    """

    def __init__(self):
        super(Glasso, self).__init__()

    def predict(self, data, **kwargs):
        """

        :param data: raw data df
        :param kwargs: alpha hyper-parameter (
        :return:
        """
        alpha = kwargs.get('alpha', 0.01)
        max_iter = kwargs.get('max_iter', 2000)
        edge_model = GraphLasso(alpha=alpha, max_iter=max_iter)
        edge_model.fit(data.as_matrix())

        return nx.relabel_nodes(nx.DiGraph(edge_model.get_precision()),
                                {idx: i for idx, i in enumerate(data.columns)})


class RandomizedLasso_model(FeatureSelectionModel):
    """RandomizedLasso from scikit-learn."""

    def __init__(self):
        super(RandomizedLasso_model, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        alpha = kwargs.get("alpha", 'aic')
        scaling = kwargs.get("scaling", 0.5)
        sample_fraction = kwargs.get("sample_fraction", 0.75)
        n_resampling = kwargs.get("n_resampling", 10)

        randomized_lasso = RandomizedLasso(alpha=alpha, scaling=scaling, sample_fraction=sample_fraction,
                                           n_resampling=n_resampling)
        randomized_lasso.fit(df_features.as_matrix(), np.ravel(df_target.as_matrix()))

        return randomized_lasso.scores_


class HSICLasso(FeatureSelectionModel):
    def __init__(self):
        super(HSICLasso, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):

        y = np.transpose(df_target.as_matrix())
        X = np.transpose(df_features.as_matrix())

        path, beta, A, lam = hsiclasso(X, y)

        return beta
