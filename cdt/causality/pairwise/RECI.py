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
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
import numpy as np
from .model import PairwiseModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class RECI(PairwiseModel):
    """
    RECI, Best-fit mse with monome regressor and [0,1] rescaling

    Args:
        degree (int): Degree of the polynomial regression.

    .. note::
       Bloebaum, P., Janzing, D., Washio, T., Shimizu, S., & Schoelkopf, B.
       (2018, March). Cause-Effect Inference by Comparing Regression Errors.
       In International Conference on Artificial Intelligence and Statistics (pp. 900-909).
    """
    def __init__(self, degree=3):
        super(RECI, self).__init__()
        self.degree = degree

    def predict_proba(self, a, b, **kwargs):
        """ Infer causal relationships between 2 variables using the RECI statistic

        :param a: Input variable 1
        :param b: Input variable 2
        :return: Causation coefficient (Value : 1 if a->b and -1 if b->a)
        :rtype: float
        """
        return self.b_fit_score(b, a) - self.b_fit_score(a, b)

    def b_fit_score(self, x, y):
        """ Compute the RECI fit score

        Args:
            x (numpy.ndarray): Variable 1
            y (numpy.ndarray): Variable 2

        Returns:
            float: RECI fit score

        """
        x = np.reshape(minmax_scale(x), (-1, 1))
        y = np.reshape(minmax_scale(y), (-1, 1))
        poly = PolynomialFeatures(degree=self.degree)
        poly_x = poly.fit_transform(x)

        poly_x[:,1] = 0
        poly_x[:,2] = 0

        regressor = LinearRegression()
        regressor.fit(poly_x, y)

        y_predict = regressor.predict(poly_x)
        error = mean_squared_error(y_predict, y)

        return error
