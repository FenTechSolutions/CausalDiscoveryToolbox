"""
Pairwise causal models base class
Author: Olivier Goudet
Date : 7/06/2017
"""


class DeconvolutionModel(object):
    """ Base class for all graphical_models models"""

    def __init__(self):
        """ Init. """
        super(DeconvolutionModel, self).__init__()

    def predict(self, df_data):
        """ get the skeleton of the graph from raw data
        :param df_data: data to construct a graph from
        """
        return self.create_skeleton_from_data(df_data)

    def create_skeleton_from_data(self, data):

        raise NotImplementedError
