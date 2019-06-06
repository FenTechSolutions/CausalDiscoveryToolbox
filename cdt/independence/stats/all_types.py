"""Dependency criteria covering all types (Numerical, Categorical, Binary).

Author: Diviyan Kalainathan
Date: 1/06/2017

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

import sklearn.metrics as metrics
import numpy as np
from .model import IndependenceModel


def bin_variable(var, bins='fd'):  # bin with normalization
    """Bin variables w/ normalization."""
    var = np.array(var).astype(np.float)
    var = (var - np.mean(var)) / np.std(var)
    var = np.digitize(var, np.histogram(var, bins=bins)[1])

    return var


class AdjMI(IndependenceModel):
    """Dependency criterion made of binning and mutual information.

    The dependency metric relies on using the clustering metric adjusted mutual information applied
    to binned variables using the Freedman Diaconis Estimator.

    .. note::
       Ref: Vinh, Nguyen Xuan and Epps, Julien and Bailey, James, "Information theoretic measures for clusterings
       comparison: Variants, properties, normalization and correction for chance", Journal of Machine Learning
       Research, Volume 11, Oct 2010.
       Ref: Freedman, David and Diaconis, Persi, "On the histogram as a density estimator:L2 theory",
       "Zeitschrift für Wahrscheinlichkeitstheorie und Verwandte Gebiete", 1981, issn=1432-2064,
       doi=10.1007/BF01025868.
       
   Example:
       >>> from cdt.independence.stats import AdjMI
       >>> obj = AdjMI()
       >>> a = np.array([1, 2, 1, 5])
       >>> b = np.array([1, 3, 0, 6])
       >>> obj.predict(a, b)
    """

    def __init__(self):
        """Init the model."""
        super(AdjMI, self).__init__()

    def predict(self, a, b, **kwargs):
        """Perform the independence test.

        :param a: input data
        :param b: input data
        :type a: array-like, numerical data
        :type b: array-like, numerical data
        :return: dependency statistic (1=Highly dependent, 0=Not dependent)
        :rtype: float
        """
        binning_alg = kwargs.get('bins', 'fd')
        return metrics.adjusted_mutual_info_score(bin_variable(a, bins=binning_alg),
                                                  bin_variable(b, bins=binning_alg))


class NormMI(IndependenceModel):
    """Dependency criterion made of binning and mutual information.

    The dependency metric relies on using the clustering metric adjusted mutual information applied
    to binned variables using the Freedman Diaconis Estimator.
    :param a: input data
    :param b: input data
    :type a: array-like, numerical data
    :type b: array-like, numerical data
    :return: dependency statistic (1=Highly dependent, 0=Not dependent)
    :rtype: float

    .. note::
       Ref: Vinh, Nguyen Xuan and Epps, Julien and Bailey, James, "Information theoretic measures for clusterings
       comparison: Variants, properties, normalization and correction for chance", Journal of Machine Learning
       Research, Volume 11, Oct 2010.
       Ref: Freedman, David and Diaconis, Persi, "On the histogram as a density estimator:L2 theory",
       "Zeitschrift für Wahrscheinlichkeitstheorie und Verwandte Gebiete", 1981, issn=1432-2064,
       doi=10.1007/BF01025868.
       
    Example:
        >>> from cdt.independence.stats import NormMI
        >>> obj = NormMI()
        >>> a = np.array([1, 2, 1, 5])
        >>> b = np.array([1, 3, 0, 6])
        >>> obj.predict(a, b)

    """

    def __init__(self):
        """Init the model."""
        super(NormMI, self).__init__()

    def predict(self, a, b, **kwargs):
        """Perform the independence test.

        :param a: input data
        :param b: input data
        :type a: array-like, numerical data
        :type b: array-like, numerical data
        :return: dependency statistic (1=Highly dependent, 0=Not dependent)
        :rtype: float
        """
        binning_alg = kwargs.get('bins', 'fd')
        return metrics.adjusted_mutual_info_score(bin_variable(a, bins=binning_alg),
                                                  bin_variable(b, bins=binning_alg))
