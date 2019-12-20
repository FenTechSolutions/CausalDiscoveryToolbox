"""Build undirected graph out of raw data.

Author: Diviyan Kalainathan
Date: 1/06/17

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.

"""
import networkx as nx
from sklearn.covariance import GraphicalLasso
from .model import GraphSkeletonModel, FeatureSelectionModel
from .HSICLasso import hsiclasso
import numpy as np


class Glasso(GraphSkeletonModel):
    """Graphical Lasso to find an adjacency matrix

    .. note::
       Ref : Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
       covariance estimation with the graphical lasso. Biostatistics, 9(3),
       432-441.
       
   Example:
       >>> from cdt.independence.graph import Glasso
       >>> df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
       >>> obj = Glasso()
       >>> output = obj.predict(df)
    """

    def __init__(self):
        super(Glasso, self).__init__()

    def predict(self, data, alpha=0.01, max_iter=2000, **kwargs):
        """ Predict the graph skeleton.

        Args:
            data (pandas.DataFrame): observational data
            alpha (float): regularization parameter
            max_iter (int): maximum number of iterations

        Returns:
            networkx.Graph: Graph skeleton
        """
        edge_model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
        edge_model.fit(data.values)

        return nx.relabel_nodes(nx.DiGraph(edge_model.get_precision()),
                                {idx: i for idx, i in enumerate(data.columns)})


class HSICLasso(FeatureSelectionModel):
    """Graphical Lasso with a kernel-based independence test.
        
        Example:
            >>> from cdt.independence.graph import HSICLasso
            >>> from sklearn.datasets import load_boston
            >>> boston = load_boston()
            >>> df_features = pd.DataFrame(boston['data'])
            >>> df_target = pd.DataFrame(boston['target'])
            >>> obj = HSICLasso()
            >>> output = obj.predict_features(df_features, df_target)
            >>> ugraph = obj.predict(df_features)  # Predict skeleton
    
    """
    def __init__(self):
        super(HSICLasso, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        """For one variable, predict its neighbouring nodes.

        Args:
            df_features (pandas.DataFrame):
            df_target (pandas.Series):
            idx (int): (optional) for printing purposes
            kwargs (dict): additional options for algorithms

        Returns:
            list: scores of each feature relatively to the target

        """

        y = np.transpose(df_target.values)
        X = np.transpose(df_features.values)

        path, beta, A, lam = hsiclasso(X, y)

        return beta
