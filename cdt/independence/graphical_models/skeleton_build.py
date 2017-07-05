"""
Build undirected graph out of raw data
Author: Diviyan Kalainathan
Date: 1/06/17

"""
import numpy as np
from sklearn.covariance import GraphLasso

def skeleton_glasso(df):
    """Apply Glasso to find an adjacency matrix

    Ref : ToDo - P.Buhlmann

    :param df: Raw data table
    :type df: pandas.DataFrame
    :return: Skeleton matrix - undirected graph
    :rtype: UndirectedGraph
    """

    edge_model = GraphLasso(alpha=0.01, max_iter=2000)
    edge_model.fit(df.as_matrix())
    return edge_model.get_precision()
