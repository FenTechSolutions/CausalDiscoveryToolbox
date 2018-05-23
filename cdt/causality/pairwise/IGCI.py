"""Information Geometric Causal Inference (IGCI) model.

P. Daniušis, D. Janzing, J. Mooij, J. Zscheischler, B. Steudel,
K. Zhang, B. Schölkopf:  Inferring deterministic causal relations.
Proceedings of the 26th Annual Conference on Uncertainty in Artificial  Intelligence (UAI-2010).
http://event.cwi.nl/uai2010/papers/UAI2010_0121.pdf

Adapted by Diviyan Kalainathan
"""

from .model import PairwiseModel
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
from scipy.special import psi
import numpy as np

min_max_scale = MinMaxScaler()
standard_scale = StandardScaler()


def eval_entropy(x):
    """Evaluate the entropy of the input variable.

    :param x: input variable 1D
    :return: entropy of x
    """
    hx = 0.
    sx = sorted(x)
    for i, j in zip(sx[:-1], sx[1:]):
        delta = j-i
        if bool(delta):
            hx += np.log(np.abs(delta))
    hx = hx / (len(x) - 1) + psi(len(x)) - psi(1)

    return hx


def integral_approx_estimator(x, y):
    """Integral approximation estimator for causal inference.

    :param x: input variable x 1D
    :param y: input variable y 1D
    :return: Return value of the IGCI model >0 if x->y otherwise if return <0
    """
    a, b = (0., 0.)
    x = np.array(x)
    y = np.array(y)
    idx, idy = (np.argsort(x), np.argsort(y))

    for x1, x2, y1, y2 in zip(x[[idx]][:-1], x[[idx]][1:], y[[idx]][:-1], y[[idx]][1:]):
        if x1 != x2 and y1 != y2:
            a = a + np.log(np.abs((y2 - y1) / (x2 - x1)))

    for x1, x2, y1, y2 in zip(x[[idy]][:-1], x[[idy]][1:], y[[idy]][:-1], y[[idy]][1:]):
        if x1 != x2 and y1 != y2:
            b = b + np.log(np.abs((x2 - x1) / (y2 - y1)))

    return (a - b)/len(x)


class IGCI(PairwiseModel):
    """Information Geometric Causal Inference (IGCI) model.

    P. Daniušis, D. Janzing, J. Mooij, J. Zscheischler, B. Steudel,
    K. Zhang, B. Schölkopf:  Inferring deterministic causal relations.
    Proceedings of the 26th Annual Conference on Uncertainty in Artificial  Intelligence (UAI-2010).
    http://event.cwi.nl/uai2010/papers/UAI2010_0121.pdf
    """

    def __init__(self):
        """.Initialize the IGCI model."""
        super(IGCI, self).__init__()

    def predict_proba(self, a, b, **kwargs):
        """Evaluate a pair using the IGCI model.

        :param a: Input variable 1D
        :param b: Input variable 1D
        :param kwargs: {refMeasure: Scaling method (gaussian, integral or None),
                        estimator: method used to evaluate the pairs (entropy or integral)}
        :return: Return value of the IGCI model >0 if a->b otherwise if return <0
        """
        estimators = {'entropy': lambda x, y: eval_entropy(y) - eval_entropy(x), 'integral': integral_approx_estimator}
        ref_measures = {'gaussian': lambda x: standard_scale.fit_transform(x.reshape((-1, 1))),
                        'uniform': lambda x: min_max_scale.fit_transform(x.reshape((-1, 1))), 'None': lambda x: x}

        ref_measure = ref_measures[kwargs.get('refMeasure', 'gaussian')]
        estimator = estimators[kwargs.get('estimator', 'entropy')]

        a = ref_measure(a)
        b = ref_measure(b)

        return estimator(a, b)
