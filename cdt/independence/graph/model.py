"""Graph Skeleton Recovery models base class.

Author: Diviyan Kalainathan
Date : 7/06/2017
"""
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ...utils.Settings import SETTINGS


class GraphSkeletonModel(object):
    """Base class for undirected graph recovery."""

    def __init__(self):
        """Init the model."""
        super(GraphSkeletonModel, self).__init__()

    def predict(self, data):
        """Infer a undirected graph out of data."""
        raise NotImplementedError


class FeatureSelectionModel(GraphSkeletonModel):
    """Base class for methods using feature selection on each variable."""

    def __init__(self):
        """Init the model."""
        super(FeatureSelectionModel, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        """For one variable, predict its neighbours."""
        raise NotImplementedError

    def run_feature_selection(self, df_data, target, idx, **kwargs):
        """Run feature selection for one node."""
        list_features = list(df_data.columns.values)
        list_features.remove(target)
        df_target = pd.DataFrame(df_data[target], columns=[target])
        df_features = df_data[list_features]

        return self.predict_features(df_features, df_target, **kwargs)

    def predict(self, df_data, threshold=0.05, **kwargs):
        """Get the skeleton of the graph from raw data.

        :param df_data: data to construct a graph from
        """
        nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
        list_nodes = list(df_data.columns.values)
        if nb_jobs != 1:
            result_feature_selection = Parallel(n_jobs=nb_jobs)(delayed(self.run_feature_selection)
                                                                (df_data, node, idx, **kwargs)
                                                                for idx, node in enumerate(list_nodes))
        else:
            result_feature_selection = [self.run_feature_selection(df_data, node, idx, **kwargs) for idx, node in enumerate(list_nodes)]
        for idx, i in enumerate(result_feature_selection):
            try:
                i.insert(idx, 0)
            except AttributeError:  # if results are numpy arrays
                result_feature_selection[idx] = np.insert(i, idx, 0)
        matrix_results = np.array(result_feature_selection)
        matrix_results *= matrix_results.transpose()
        np.fill_diagonal(matrix_results, 0)
        matrix_results /= 2

        graph = nx.Graph()

        for (i, j), x in np.ndenumerate(matrix_results):
            if matrix_results[i, j] > threshold:
                graph.add_edge(list_nodes[i], list_nodes[j],
                               weight=matrix_results[i, j])
        for node in list_nodes:
            if node not in graph.nodes():
                graph.add_node(node)
        return graph
