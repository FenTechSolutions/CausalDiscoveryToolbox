"""Dependency criteria covering all types (Numerical, Categorical, Binary).

Author: Diviyan Kalainathan
Date: 1/06/2017

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
    Ref: Vinh, Nguyen Xuan and Epps, Julien and Bailey, James, "Information theoretic measures for clusterings
        comparison: Variants, properties, normalization and correction for chance", Journal of Machine Learning
        Research, Volume 11, Oct 2010.
    Ref: Freedman, David and Diaconis, Persi, "On the histogram as a density estimator:L2 theory",
        "Zeitschrift für Wahrscheinlichkeitstheorie und Verwandte Gebiete", 1981, issn=1432-2064,
        doi=10.1007/BF01025868.
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
    Ref: Vinh, Nguyen Xuan and Epps, Julien and Bailey, James, "Information theoretic measures for clusterings
        comparison: Variants, properties, normalization and correction for chance", Journal of Machine Learning
        Research, Volume 11, Oct 2010.
    Ref: Freedman, David and Diaconis, Persi, "On the histogram as a density estimator:L2 theory",
        "Zeitschrift für Wahrscheinlichkeitstheorie und Verwandte Gebiete", 1981, issn=1432-2064,
        doi=10.1007/BF01025868.

    :param a: input data
    :param b: input data
    :type a: array-like, numerical data
    :type b: array-like, numerical data
    :return: dependency statistic (1=Highly dependent, 0=Not dependent)
    :rtype: float
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
