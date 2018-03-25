"""Graph Skeleton Recovery models base class.

Author: Diviyan Kalainathan
Date : 7/06/2017
"""


class GraphSkeletonModel(object):
    """Base class for undirected graph recovery."""

    def __init__(self):
        """Init the model."""
        super(GraphSkeletonModel, self).__init()

    def predict(self, data):
        """Infer a undirected graph out of data."""
        raise NotImplementedError
