"""
Pairwise causal models base class
Author: Olivier Goudet
Date : 7/06/2017
"""
from ...utils.Settings import SETTINGS
from joblib import Parallel, delayed
import numpy as np
from ...utils.Graph import UndirectedGraph


class DeconvolutionModel(object):
    """ Base class for all graphs models"""

    def __init__(self):
        """ Init. """
        super(DeconvolutionModel, self).__init__()

    def predict(self, df_data, **kwargs):
        """ get the skeleton of the graph from raw data
        :param df_data: data to construct a graph from
        """
        return self.create_skeleton_from_data(df_data, **kwargs)

    def create_skeleton_from_data(self, data, **kwargs):
        raise NotImplementedError


class FeatureSelectionModel(object):
    """ Base class for all graphs models"""

    def __init__(self):
        """ Init. """
        super(FeatureSelectionModel, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        """ get the relevance score of each candidate variable with feature selection method
        :param df_features: candidate feature variables
        :param df_target: target variable
        """
        raise NotImplementedError

    def run_feature_selection(self, df_data, target, idx, **kwargs):

        list_features = list(df_data.columns.values)
        list_features.remove(target)

        df_target = df_data[target]
        df_features = df_data[list_features]

        scores = self.predict_features(df_features, df_target, idx, **kwargs)

        return scores

    def predict(self, df_data, **kwargs):
        """ get the skeleton of the graph from raw data
        :param df_data: data to construct a graph from
        """
        nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)

        list_nodes = list(df_data.columns.values)
        n_nodes = len(list_nodes)
        matrix_results = np.zeros((n_nodes, n_nodes))

        result_feature_selection = Parallel(n_jobs=nb_jobs)(
            delayed(self.run_feature_selection)(df_data, node, idx, **kwargs) for idx, node in enumerate(list_nodes))

        for i in range(len(result_feature_selection)):

            score = result_feature_selection[i]
            cpt = 0

            for j in range(len(list_nodes)):
                if (j != i):
                    matrix_results[i, j] = matrix_results[i, j] + score[cpt]
                    matrix_results[j, i] = matrix_results[j, i] + score[cpt]
                    cpt += 1

        matrix_results = matrix_results / 2

        graph = UndirectedGraph()

        for i in range(n_nodes):
            for j in range(n_nodes):
                if (j > i):
                    graph.add(df_data.columns.values[i], df_data.columns.values[j], matrix_results[i, j])

        return graph
