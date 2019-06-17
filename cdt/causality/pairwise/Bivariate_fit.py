"""
Bivariate fit model
Author : Olivier Goudet
Date : 7/06/17

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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
import numpy as np
from .model import PairwiseModel
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3)
from sklearn.linear_model import LinearRegression


class BivariateFit(PairwiseModel):
    """Bivariate Fit model.

    **Description:** The bivariate fit model is  based onon a best-fit criterion
    relying on a Gaussian Process regressor. Used as weak baseline.

    **Data Type**: Continuous

    **Assumptions**: This is often a model used to show that correlation
    :math:`\\neq` causation. It holds very weak performance, as it states that
    the best predictive model is the causal model.

    Example:
        >>> from cdt.causality.pairwise import BivariateFit
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from cdt.data import load_dataset
        >>> data, labels = load_dataset('tuebingen')
        >>> obj = BivariateFit()
        >>>
        >>> # This example uses the predict() method
        >>> output = obj.predict(data)
        >>>
        >>> # This example uses the orient_graph() method. The dataset used
        >>> # can be loaded using the cdt.data module
        >>> data, graph = load_dataset("sachs")
        >>> output = obj.orient_graph(data, nx.Graph(graph))
        >>>
        >>> #To view the directed graph run the following command
        >>> nx.draw_networkx(output, font_size=8)
        >>> plt.show()
    """
    def __init__(self, ffactor=2, maxdev=3, minc=12):
        super(BivariateFit, self).__init__()

    def predict_proba(self, dataset, **kwargs):
        """ Infer causal relationships between 2 variables using regression.

        Args:
            dataset (tuple): Couple of np.ndarray variables to classify

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        a, b = dataset
        return self.b_fit_score(b, a) - self.b_fit_score(a, b)

    def b_fit_score(self, x, y):
        """ Computes the cds statistic from variable 1 to variable 2

        Args:
            a (numpy.ndarray): Variable 1
            b (numpy.ndarray): Variable 2

        Returns:
            float: BF fit score
        """
        x = np.reshape(scale(x), (-1, 1))
        y = np.reshape(scale(y), (-1, 1))
        gp = GaussianProcessRegressor().fit(x, y)
        y_predict = gp.predict(x)
        error = mean_squared_error(y_predict, y)

        return error
