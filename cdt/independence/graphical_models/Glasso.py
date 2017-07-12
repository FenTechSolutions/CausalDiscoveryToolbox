"""
Build undirected graph out of raw data
Author: Diviyan Kalainathan
Date: 1/06/17

"""
import numpy as np
from sklearn.covariance import GraphLasso
from .model import DeconvolutionModel


class Glasso(DeconvolutionModel):
    """Apply Glasso to find an adjacency matrix

    Ref : ToDo - P.Buhlmann

    :param df: Raw data table
    :type df: pandas.DataFrame
    :return: Skeleton matrix - undirected graph
    :rtype: UndirectedGraph
    """
    def __init__(self):
        super(Glasso, self).__init__()

    def create_skeleton_from_data(self, data, **kwargs):
        alpha = kwargs.get('alpha', 0.01)
        edge_model = GraphLasso(alpha=alpha, max_iter=2000)
        edge_model.fit(data.as_matrix())
        return edge_model.get_precision()
