"""
Build undirected graph out of raw data
Author: Diviyan Kalainathan
Date: 1/06/17

"""
import numpy as np
from sklearn.covariance import GraphLasso

def skeleton_glasso(df):
    """Apply Lasso CV to find an adjacency matrix

    :param df: Raw data table
    :type df: pandas.DataFrame
    :return: Skeleton matrix - undirected graph
    :rtype: UndirectedGraph
    """

    edge_model = GraphLasso(alpha=0.01, max_iter=2000)
    edge_model.fit(df.as_matrix())
    return edge_model.get_precision()


def build_skeleton_pairwise(df, criterion):
    """ Build a skeleton using a pairwise dependence criterion

    :param df: Raw data table
    :param criterion: Pairwise dependence criterion
    :type df: pandas.DataFrame
    :type criterion: function
    :return: Skeleton matrix - undirected graph
    :rtype: UndirectedGraph
    """

    nb_var = len(df.columns)
    skeleton = np.ones((nb_var, nb_var))
    col = df.columns
    for i in range(nb_var):
        for j in range(i, nb_var):
            skeleton[i, j] = criterion(
                list(df[df.columns[i]]), list(df[df.columns[j]]))
            skeleton[j, i] = skeleton[i, j]

    return skeleton
