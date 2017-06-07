"""
Pairwise causal models base class
Author: Diviyan Kalainathan
Date : 7/06/2017
"""
from sklearn.preprocessing import scale

class Pairwise_Model(object):
    """ Base class for all pairwise causal inference models

    Usage for undirected/directed graphs and CEPC df format.
    """
    def __init__(self):
        """ Init. """
        super(Pairwise_Model, self).__init__()

    def predictor(self, a, b):
        """ Prediction method for pairwise causal inference.
        Predictor is meant to be overridden in all subclasses

        :param a: Variable 1
        :param b: Variable 2
        :return: probability (Value : 1 if a->b and -1 if b->a)
        :rtype: float
        """
        raise NotImplementedError

    def predict_proba(self, x):
        """ Causal prediction of a pairwise dataset (x,y)

        :param x: Pairwise dataset
        :type x: cepc_df format
        :return: predictions probabilities
        :rtype: list
        """

        pred = []
        for idx, row in x.iterrows():
            a = scale(row['A'].reshape((len(row['A']), 1)))
            b = scale(row['B'].reshape((len(row['B']), 1)))

            pred.append(self.predictor(a, b))
        return pred

    def orient_graph(self, x, df_data):
        """ Orient an undirected graph using the pairwise method defined by the subclass

        :param x:
        :param df_data:
        :return: Directed graph w/ weights
        """
        # ToDo: Implement !
        pass
