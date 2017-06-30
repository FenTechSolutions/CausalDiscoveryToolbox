"""
Pairwise causal models base class
Author: Diviyan Kalainathan
Date : 7/06/2017
"""



class DeconvolutionModel(object):
    """ Base class for all deconvolution model"""


    def __init__(self):
        """ Init. """
        super(DeconvolutionModel, self).__init__()

    def predict(self, df_data):
        """ get the skeleton of the graph from raw data
        :param df_data:
        """
        return self.create_skeleton_from_data(df_data)

    def create_skeleton_from_data(self, data):

        raise NotImplementedError




