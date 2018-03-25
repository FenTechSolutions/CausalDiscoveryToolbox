"""Dependency criteria for Numerical data.

Author: Diviyan Kalainathan
Date: 1/06/2017

"""

import numpy as np
import scipy.stats as sp
from .model import IndependenceModel
from sklearn.feature_selection import mutual_info_regression


def rbf_dot2(p1, p2, deg):
    if p1.ndim == 1:
        p1 = p1[:, np.newaxis]
        p2 = p2[:, np.newaxis]

    size1 = p1.shape
    size2 = p2.shape

    G = np.sum(p1 * p1, axis=1)[:, np.newaxis]
    H = np.sum(p2 * p2, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))
    H = Q + R - 2.0 * np.dot(p1, p2.T)
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def rbf_dot(X, deg):
    # Set kernel size to median distance between points, if no kernel specified
    if X.ndim == 1:
        X = X[:, np.newaxis]
    m = X.shape[0]
    G = np.sum(X * X, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, m))
    H = Q + Q.T - 2.0 * np.dot(X, X.T)
    if deg == -1:
        dists = (H - np.tril(H)).flatten()
        deg = np.sqrt(0.5 * np.median(dists[dists > 0]))
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def FastHsicTestGamma(X, Y, sig=[-1, -1], maxpnt=200):
    """This function implements the HSIC independence test using a Gamma approximation
     to the test threshold. Use at most maxpnt points to save time.

    :param X: contains dx columns, m rows. Each row is an i.i.d sample
    :param Y: contains dy columns, m rows. Each row is an i.i.d sample
    :param sig: [0] (resp [1]) is kernel size for x(resp y) (set to median distance if -1)
    :return: test statistic

    """

    m = X.shape[0]
    if m > maxpnt:
        indx = np.floor(np.r_[0:m:float(m - 1) / (maxpnt - 1)]).astype(int)
        #       indx = np.r_[0:maxpnt]
        Xm = X[indx].astype(float)
        Ym = Y[indx].astype(float)
        m = Xm.shape[0]
    else:
        Xm = X.astype(float)
        Ym = Y.astype(float)

    H = np.eye(m) - 1.0 / m * np.ones((m, m))

    K = rbf_dot(Xm, sig[0])
    L = rbf_dot(Ym, sig[1])

    Kc = np.dot(H, np.dot(K, H))
    Lc = np.dot(H, np.dot(L, H))

    testStat = (1.0 / m) * (Kc.T * Lc).sum()
    if ~np.isfinite(testStat):
        testStat = 0

    return testStat


class PearsonCorrelation(IndependenceModel):
    def __init__(self):
        super(PearsonCorrelation, self).__init__()

    def predict(self, a, b):
        return sp.pearsonr(a, b)[0]


class SpearmanCorrelation(IndependenceModel):
    def __init__(self):
        super(SpearmanCorrelation, self).__init__()

    def predict(self, a, b):
        return sp.spearmanr(a, b)[0]


class MIRegression(IndependenceModel):
    def __init__(self):
        super(MIRegression, self).__init__()

    def predict(self, a, b):
        a = np.array(a).reshape((-1, 1))
        b = np.array(b).reshape((-1, 1))
        return (mutual_info_regression(a, b.reshape((-1,))) + mutual_info_regression(b, a.reshape((-1,))))/2


class KendallTau(IndependenceModel):
    def __init__(self):
        super(KendallTau, self).__init__()

    def predict(self, a, b):
        a = np.array(a).reshape((-1, 1))
        b = np.array(b).reshape((-1, 1))
        return sp.kendalltau(a, b)[0]


class NormalizedHSIC(IndependenceModel):
    def __init__(self):
        super(NormalizedHSIC, self).__init__()

    def predict(self, a, b, sig=[-1, -1], maxpnt=500):
        a = (a - np.mean(a)) / np.std(a)
        b = (b - np.mean(b)) / np.std(b)

        return FastHsicTestGamma(a, b, sig, maxpnt)
