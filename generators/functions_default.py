"""
Basic functions for causal generation
Author : David Lopez-Paz, Facebook AI Research, modified by Diviyan Kalainathan
"""

import numpy as np
from sklearn.preprocessing import scale
from scipy.interpolate import UnivariateSpline as sp
import warnings
import random

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.mixture import GMM


def cause(n, k=4, p1=2, p2=2):
    g = GMM(k)
    g.means_ = p1 * np.random.randn(k, 1)
    g.covars_ = np.power(abs(p2 * np.random.randn(k, 1) + 1), 2)
    g.weights_ = abs(np.random.rand(k, 1))
    g.weights_ = g.weights_ / sum(g.weights_)
    # return scale(g.sample(n)).flatten()
    return np.random.uniform(-1, 1, n)
    # return g.sample(n).flatten()


def noise(n, v):
    return v * np.random.rand(1) * np.random.randn(n, 1) + random.sample([2, -2], 1)
    # np.random.randint(-1,1)


def mechanism(x, d):
    g = np.linspace(min(x) - np.std(x), max(x) + np.std(x), d);
    return sp(g, np.random.randn(d))(x.flatten())[:, np.newaxis]


def effect(x, n, v, d=4):
    y = np.array(x)
    # return scale(scale(mechanism(y,d))+noise(n,v)).flatten()
    return scale(mechanism(y, d)).flatten()


def rand_bin(x):
    numCat1 = np.random.randint(2, 20)
    maxstd = 3
    x = scale(x)
    bins = np.linspace(-maxstd, maxstd, num=numCat1 + 1)
    x = np.digitize(x, bins) - 1

    return x
