"""
Feature extraction

"""

# Author: Jose A. R. Fonollosa <jarfo@yahoo.com>
#
# License: Apache, Version 2.0

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import adjusted_mutual_info_score
from scipy.special import psi
from scipy.stats.stats import pearsonr
from scipy.stats import skew, kurtosis
from collections import Counter, defaultdict
from multiprocessing import Pool
import pandas as pd
import operator
from .hsic import FastHsicTestGamma
import math

BINARY = "Binary"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"


class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name in self.features:
            extractor.fit(X[feature_name].values[:, np.newaxis], y)

    def transform(self, X):
        return X[self.features].values

    def fit_transform(self, X, y=None):
        return self.transform(X)


def weighted_mean_and_std(values, weights):
    """
    Returns the weighted average and standard deviation.

    values, weights -- numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=0)
    variance = np.dot(weights, (values - average) ** 2) / weights.sum()  # Fast and numerically precise
    return (average, np.sqrt(variance))


def count_unique(x):
    try:
        return len(set(x))
    except TypeError:
        return len(set(x.flat))


def count_unique_ratio(x):
    try:
        return len(set(x)) / float(len(x))
    except TypeError:
        return len(set(x.flat))/float(len(x))


def binary(tp):
    assert type(tp) is str
    return tp == BINARY


def categorical(tp):
    assert type(tp) is str
    return tp == CATEGORICAL


def numerical(tp):
    assert type(tp) is str
    return tp == NUMERICAL


def binary_entropy(p, base):
    assert p <= 1 and p >= 0
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p)) if (p != 0) and (p != 1) else 0
    return h / np.log(base)


def discrete_probability(x, tx, ffactor, maxdev):
    x = discretized_sequence(x, tx, ffactor, maxdev)
    try:
        return Counter(x)
    except TypeError as e:
        return Counter(np.array(x).flat) if isinstance(x, list) else Counter(x.flat)


def discretized_values(x, tx, ffactor, maxdev):
    if numerical(tx) and count_unique(x) > (2 * ffactor * maxdev + 1):
        vmax = ffactor * maxdev
        vmin = -ffactor * maxdev
        return range(vmin, vmax + 1)
    else:
        try:
            return sorted(list(set(x)))
        except TypeError:
            return sorted(list(set(x.flat)))


def len_discretized_values(x, tx, ffactor, maxdev):
    return len(discretized_values(x, tx, ffactor, maxdev))


def discretized_sequence(x, tx, ffactor, maxdev, norm=True):
    if not norm or (numerical(tx) and count_unique(x) > len_discretized_values(x, tx, ffactor, maxdev)):
        if norm:
            x = (x - np.mean(x)) / np.std(x)
            xf = x[abs(x) < maxdev]
            x = (x - np.mean(xf)) / np.std(xf)
        x = np.round(x * ffactor)
        vmax = ffactor * maxdev
        vmin = -ffactor * maxdev
        x[x > vmax] = vmax
        x[x < vmin] = vmin
    return x


def discretized_sequences(x, tx, y, ty, ffactor=3, maxdev=3):
    return discretized_sequence(x, tx, ffactor, maxdev), discretized_sequence(y, ty, ffactor, maxdev)


def normalized_error_probability(x, tx, y, ty, ffactor=3, maxdev=3):
    x, y = discretized_sequences(x, tx, y, ty, ffactor, maxdev)
    try:
        cx = Counter(x)
        cy = Counter(y)
    except TypeError:
        cx = Counter(x.flat)
        cy = Counter(y.flat)
    
    nx = len(cx)
    ny = len(cy)
    pxy = defaultdict(lambda: 0)
    try:
        for p in zip(x, y):
            pxy[p] += 1
    except TypeError:
        for p in zip(x.flat, y.flat):
            pxy[p] += 1
    pxy = np.array([[pxy[(a, b)] for b in cy] for a in cx], dtype=float)
    pxy = pxy / pxy.sum()
    perr = 1 - np.sum(pxy.max(axis=1))
    max_perr = 1 - np.max(pxy.sum(axis=0))
    pnorm = perr / max_perr if max_perr > 0 else perr
    return pnorm


def discrete_entropy(x, tx, ffactor=3, maxdev=3, bias_factor=0.7):
    c = discrete_probability(x, tx, ffactor, maxdev)
    # print(c, len(c))
    pk = np.array(list(c.values()), dtype=float)
    pk = pk / pk.sum()
    vec = pk * np.log(pk)
    S = -np.sum(vec, axis=0)
    return S + bias_factor * (len(pk) - 1) / float(2 * len(list(x)))


def discrete_divergence(cx, cy):
    for a, v in cx.most_common():
        if cy[a] == 0:
            cy[a] = 1
    nx = float(sum(cx.values()))
    ny = float(sum(cy.values()))
    sum = 0.
    for a, v in cx.most_common():
        px = v / nx
        py = cy[a] / ny
        sum += px * np.log(px / py)
    return sum


def discrete_joint_entropy(x, tx, y, ty, ffactor=3, maxdev=3):
    x, y = discretized_sequences(x, tx, y, ty, ffactor, maxdev)
    return discrete_entropy(list(zip(x, y)), CATEGORICAL)


def normalized_discrete_joint_entropy(x, tx, y, ty, ffactor=3, maxdev=3):
    x, y = discretized_sequences(x, tx, y, ty, ffactor, maxdev)
    e = discrete_entropy(list(zip(x, y)), CATEGORICAL)
    nx = len_discretized_values(x, tx, ffactor, maxdev)
    ny = len_discretized_values(y, ty, ffactor, maxdev)
    if nx * ny > 0: e = e / np.log(nx * ny)
    return e


def discrete_conditional_entropy(x, tx, y, ty):
    return discrete_joint_entropy(x, tx, y, ty) - discrete_entropy(y, ty)


def adjusted_mutual_information(x, tx, y, ty, ffactor=3, maxdev=3):
    x, y = discretized_sequences(x, tx, y, ty, ffactor, maxdev)
    try:
        return adjusted_mutual_info_score(x, y)
    except ValueError:
        return adjusted_mutual_info_score(x.squeeze(1), y.squeeze(1))

def discrete_mutual_information(x, tx, y, ty):
    ex = discrete_entropy(x, tx)
    ey = discrete_entropy(y, ty)
    exy = discrete_joint_entropy(x, tx, y, ty)
    mxy = max((ex + ey) - exy,
              0)  # Mutual information is always positive: max() avoid negative values due to numerical errors
    return mxy


def normalized_discrete_entropy(x, tx, ffactor=3, maxdev=3):
    e = discrete_entropy(x, tx, ffactor, maxdev)
    n = len_discretized_values(x, tx, ffactor, maxdev)
    if n > 0: e = e / np.log(n)
    return e


# Continuous information measures
def to_numerical(x, y):
    dx = defaultdict(lambda: np.zeros(2))
    for i, a in enumerate(x):
        dx[a][0] += y[i]
        dx[a][1] += 1
    for a in dx.keys():
        dx[a][0] /= dx[a][1]
    x = np.array([dx[a][0] for a in x], dtype=float)
    return x


def normalize(x, tx):
    if not numerical(tx):  # reassign labels according to its frequency
        try:
            cx = Counter(x)
        except TypeError:
            cx = Counter(x.flat)
        xmap = dict()
        # nx = len(cx)
        # center = nx/2 if (nx % 4) == 0 else (nx-1)//2
        # for i, k in enumerate(cx.most_common()):
        # offset = (i+1)//2
        # if (i % 4) > 1: offset = -offset
        # xmap[k[0]] = center + offset
        for i, k in enumerate(cx.most_common()):
            xmap[k[0]] = i
        y = np.array([xmap[a] for a in x.flat], dtype=float)
    else:
        y = x

    y = y - np.mean(y)
    if np.std(y) > 0:
        y = y / np.std(y)
    return y


def normalized_entropy_baseline(x, tx):
    try:
        if len(set(x)) < 2:
            return 0
    except TypeError:
        if len(set(x.flat)) < 2:
            return 0
    x = normalize(x, tx)
    xs = np.sort(x)
    delta = xs[1:] - xs[:-1]
    delta = delta[delta != 0]
    hx = np.mean(np.log(delta))
    hx += psi(len(delta))
    hx -= psi(1)
    return hx


def normalized_entropy(x, tx, m=2):
    x = normalize(x, tx)
    try:
        cx = Counter(x)
    except TypeError:
        cx = Counter(x.flat)

    if len(cx) < 2:
        return 0
    xk = np.array(list(cx.keys()), dtype=float)
    xk.sort()
    delta = (xk[1:] - xk[:-1]) / m
    counter = np.array([cx[i] for i in xk], dtype=float)
    hx = np.sum(counter[1:] * np.log(delta / counter[1:])) / len(x)
    hx += (psi(len(delta)) - np.log(len(delta)))
    hx += np.log(len(x))
    hx -= (psi(m) - np.log(m))
    return hx


def igci(x, tx, y, ty):
    try:
        if len(set(x)) < 2:
            return 0
    except TypeError:
        if len(set(x.flat)) < 2:
            return 0
    x = normalize(x, tx)
    y = normalize(y, ty)
    if len(x) != len(set(x.flat)):
        dx = defaultdict(lambda: np.zeros(2))
        for i, a in enumerate(x.flat):
            dx[a][0] += y[i]
            dx[a][1] += 1
        for a in dx.keys():
            dx[a][0] /= dx[a][1]
        xy = np.array(sorted([[a, dx[a][0]] for a in dx.keys()]), dtype=float)
        counter = np.array([dx[a][1] for a in xy[:, 0]], dtype=float)
    else:
        xy = np.array(sorted(zip(x, y)), dtype=float)
        counter = np.ones(len(x))
    delta = xy[1:] - xy[:-1]
    if len(delta.shape) > 2:
        delta = delta.squeeze(2)
    selec = delta[:, 1] != 0
    delta = delta[selec]
    counter = np.min([counter[1:], counter[:-1]], axis=0)
    counter = counter[selec]
    hxy = np.sum(counter * np.log(delta[:, 0] / np.abs(delta[:, 1]))) / len(x)
    return hxy


def uniform_divergence(x, tx, m=2):
    x = normalize(x, tx)
    try:
        cx = Counter(x)
    except TypeError:
        cx = Counter(x.flat)
    xk = np.array(list(cx.keys()), dtype=float)
    xk.sort()
    delta = np.zeros(len(xk))
    if len(xk) > 1:
        delta[0] = xk[1] - xk[0]
        delta[1:-1] = (xk[m:] - xk[:-m]) / m
        delta[-1] = xk[-1] - xk[-2]
    else:
        delta = np.array(np.sqrt(12))
    counter = np.array([cx[i] for i in xk], dtype=float)
    delta = delta / np.sum(delta)
    hx = np.sum(counter * np.log(counter / delta)) / len(x)
    hx -= np.log(len(x))
    hx += (psi(m) - np.log(m))
    return hx


def normalized_skewness(x, tx):
    y = normalize(x, tx)
    return skew(y)


def normalized_kurtosis(x, tx):
    y = normalize(x, tx)
    return kurtosis(y)


def normalized_moment(x, tx, y, ty, n, m):
    x = normalize(x, tx)
    y = normalize(y, ty)
    return np.mean((x ** n) * (y ** m))


def moment21(x, tx, y, ty):
    return normalized_moment(x, tx, y, ty, 2, 1)


def moment22(x, tx, y, ty):
    return normalized_moment(x, tx, y, ty, 2, 2)


def moment31(x, tx, y, ty):
    return normalized_moment(x, tx, y, ty, 3, 1)


def fit(x, tx, y, ty):
    if (not numerical(tx)) or (not numerical(ty)):
        return 0
    if (count_unique(x) <= 2) or (count_unique(y) <= 2):
        return 0
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    if len(x.shape) > 1:
        x = x.squeeze(1)
    if len(y.shape) > 1:
        y = y.squeeze(1)

    xy1 = np.polyfit(x, y, 1)
    xy2 = np.polyfit(x, y, 2)
    return abs(2 * xy2[0]) + abs(xy2[1] - xy1[0])


def fit_error(x, tx, y, ty, m=2):
    if categorical(tx) and categorical(ty):
        x = normalize(x, tx)
        y = normalize(y, ty)
    elif categorical(tx) and numerical(ty):
        x = to_numerical(x, y)
    elif numerical(tx) and categorical(ty):
        y = to_numerical(y, x)
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    if len(x.shape) > 1:
        x = x.squeeze(1)
    if len(y.shape) > 1:
        y = y.squeeze(1)
    if (count_unique(x) <= m) or (count_unique(y) <= m):
        xy = np.polyfit(x, y, min(count_unique(x), count_unique(y)) - 1)
    else:
        xy = np.polyfit(x, y, m)
    return np.std(y - np.polyval(xy, x))


def fit_noise_entropy(x, tx, y, ty, ffactor=3, maxdev=3, minc=10):
    x, y = discretized_sequences(x, tx, y, ty, ffactor, maxdev)
    try:
        cx = Counter(x)
    except TypeError:
        cx = Counter(x.flat)
    entyx = []
    for a in cx:
        if cx[a] > minc:
            entyx.append(discrete_entropy(y[x == a], CATEGORICAL))
    if len(entyx) == 0: return 0
    n = len_discretized_values(y, ty, ffactor, maxdev)
    return np.std(entyx) / np.log(n)


def fit_noise_skewness(x, tx, y, ty, ffactor=3, maxdev=3, minc=8):
    xd, yd = discretized_sequences(x, tx, y, ty, ffactor, maxdev)
    try:
        cx = Counter(xd)
    except TypeError:
        cx = Counter(xd.flat)
    skewyx = []
    for a in cx:
        if cx[a] >= minc:
            skewyx.append(normalized_skewness(y[xd == a], ty))
    if len(skewyx) == 0: return 0
    return np.std(skewyx)


def fit_noise_kurtosis(x, tx, y, ty, ffactor=3, maxdev=3, minc=8):
    xd, yd = discretized_sequences(x, tx, y, ty, ffactor, maxdev)
    try:
        cx = Counter(xd)
    except TypeError:
        cx = Counter(xd.flat)
    kurtyx = []
    for a in cx:
        if cx[a] >= minc:
            kurtyx.append(normalized_kurtosis(y[xd == a], ty))
    if len(kurtyx) == 0: return 0
    return np.std(kurtyx)


def conditional_distribution_similarity(x, tx, y, ty, ffactor=2, maxdev=3, minc=12):
    xd, yd = discretized_sequences(x, tx, y, ty, ffactor, maxdev)
    try:
        cx = Counter(xd)
        cy = Counter(yd)
    except TypeError:
        cx = Counter(xd.flat)
        cy = Counter(yd.flat)
    yrange = sorted(cy.keys())
    ny = len(yrange)
    py = np.array([cy[i] for i in yrange], dtype=float)
    py = py / py.sum()
    pyx = []
    for a in cx:
        if cx[a] > minc:
            yx = y[xd == a]
            if not numerical(ty):
                cyx = Counter(yx)
                pyxa = np.array([cyx[i] for i in yrange], dtype=float)
                pyxa.sort()
            elif count_unique(y) > len_discretized_values(y, ty, ffactor, maxdev):
                yx = (yx - np.mean(yx)) / np.std(y)
                yx = discretized_sequence(yx, ty, ffactor, maxdev, norm=False)
                cyx = Counter(yx.astype(int))
                pyxa = np.array([cyx[i] for i in discretized_values(y, ty, ffactor, maxdev)], dtype=float)
            else:
                cyx = Counter(yx)
                pyxa = [cyx[i] for i in yrange]
                pyxax = np.array([0] * (ny - 1) + pyxa + [0] * (ny - 1), dtype=float)
                xcorr = [sum(py * pyxax[i:i + ny]) for i in range(2 * ny - 1)]
                imax = xcorr.index(max(xcorr))
                pyxa = np.array([0] * (2 * ny - 2 - imax) + pyxa + [0] * imax, dtype=float)
            assert pyxa.sum() == cx[a]
            pyxa = pyxa / pyxa.sum()
            pyx.append(pyxa)

    if len(pyx) == 0: return 0
    pyx = np.array(pyx);
    pyx = pyx - pyx.mean(axis=0);
    return np.std(pyx)


def correlation(x, tx, y, ty):
    if categorical(tx) and categorical(ty):
        nperr = min(normalized_error_probability(x, tx, y, ty), normalized_error_probability(y, ty, x, tx))
        r = 1 - nperr
    else:
        if categorical(tx) and numerical(ty):
            x = to_numerical(x, y)
        elif numerical(tx) and categorical(ty):
            y = to_numerical(y, x)
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        r = np.corrcoef(x, y)[0][1]
        if np.isnan(r):
            return 0
    return r


def normalized_hsic(x, tx, y, ty):
    if categorical(tx) and categorical(ty):
        h = correlation(x, tx, y, ty)
    else:
        if categorical(tx) and numerical(ty):
            x = to_numerical(x, y)
        elif numerical(tx) and categorical(ty):
            y = to_numerical(y, x)
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        h = FastHsicTestGamma(x, y)
    return h


class SimpleTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T


class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T


def determine_type(dffeature, categorical_threshold=70):

    def type_row(feature, categorical_threshold):
        nunique_values = len(np.unique(feature))
        # print(nunique_values)
        if nunique_values < 3:
            return BINARY
        elif nunique_values < categorical_threshold:
            return CATEGORICAL
        else:
            return NUMERICAL
    return [type_row(row, categorical_threshold) for row in dffeature]


all_features = [
    ('Max', 'A', SimpleTransform(np.max)),
    ('Max', 'B', SimpleTransform(np.max)),
    ('Min', 'A', SimpleTransform(np.min)),
    ('Min', 'B', SimpleTransform(np.min)),
    ('Numerical', 'A type', SimpleTransform(numerical)),
    ('Numerical', 'B type', SimpleTransform(numerical)),
    ('Sub', ['Numerical[A type]', 'Numerical[B type]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Numerical[A type],Numerical[B type]]', SimpleTransform(abs)),

    ('Number of Samples', 'A', SimpleTransform(len)),
    ('Log', 'Number of Samples[A]', SimpleTransform(math.log)),

    ('Number of Unique Samples', 'A', SimpleTransform(count_unique)),
    ('Number of Unique Samples', 'B', SimpleTransform(count_unique)),
    ('Max', ['Number of Unique Samples[A]', 'Number of Unique Samples[B]'], MultiColumnTransform(max)),
    ('Min', ['Number of Unique Samples[A]', 'Number of Unique Samples[B]'], MultiColumnTransform(min)),
    ('Sub', ['Number of Unique Samples[A]', 'Number of Unique Samples[B]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Number of Unique Samples[A],Number of Unique Samples[B]]', SimpleTransform(abs)),

    ('Log', 'Number of Unique Samples[A]', SimpleTransform(math.log)),
    ('Log', 'Number of Unique Samples[B]', SimpleTransform(math.log)),
    ('Max', ['Log[Number of Unique Samples[A]]', 'Log[Number of Unique Samples[B]]'], MultiColumnTransform(max)),
    ('Min', ['Log[Number of Unique Samples[A]]', 'Log[Number of Unique Samples[B]]'], MultiColumnTransform(min)),
    ('Sub', ['Log[Number of Unique Samples[A]]', 'Log[Number of Unique Samples[B]]'],
     MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Log[Number of Unique Samples[A]],Log[Number of Unique Samples[B]]]', SimpleTransform(abs)),

    ('Ratio of Unique Samples', 'A', SimpleTransform(count_unique_ratio)),
    ('Ratio of Unique Samples', 'B', SimpleTransform(count_unique_ratio)),
    ('Max', ['Ratio of Unique Samples[A]', 'Ratio of Unique Samples[B]'], MultiColumnTransform(max)),
    ('Min', ['Ratio of Unique Samples[A]', 'Ratio of Unique Samples[B]'], MultiColumnTransform(min)),
    ('Sub', ['Ratio of Unique Samples[A]', 'Ratio of Unique Samples[B]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Ratio of Unique Samples[A],Ratio of Unique Samples[B]]', SimpleTransform(abs)),

    ('Normalized Entropy Baseline', ['A', 'A type'], MultiColumnTransform(normalized_entropy_baseline)),
    ('Normalized Entropy Baseline', ['B', 'B type'], MultiColumnTransform(normalized_entropy_baseline)),
    ('Max', ['Normalized Entropy Baseline[A,A type]', 'Normalized Entropy Baseline[B,B type]'],
     MultiColumnTransform(max)),
    ('Min', ['Normalized Entropy Baseline[A,A type]', 'Normalized Entropy Baseline[B,B type]'],
     MultiColumnTransform(min)),
    ('Sub', ['Normalized Entropy Baseline[A,A type]', 'Normalized Entropy Baseline[B,B type]'],
     MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Normalized Entropy Baseline[A,A type],Normalized Entropy Baseline[B,B type]]', SimpleTransform(abs)),

    ('Normalized Entropy', ['A', 'A type'], MultiColumnTransform(normalized_entropy)),
    ('Normalized Entropy', ['B', 'B type'], MultiColumnTransform(normalized_entropy)),
    ('Max', ['Normalized Entropy[A,A type]', 'Normalized Entropy[B,B type]'], MultiColumnTransform(max)),
    ('Min', ['Normalized Entropy[A,A type]', 'Normalized Entropy[B,B type]'], MultiColumnTransform(min)),
    ('Sub', ['Normalized Entropy[A,A type]', 'Normalized Entropy[B,B type]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Normalized Entropy[A,A type],Normalized Entropy[B,B type]]', SimpleTransform(abs)),

    ('IGCI', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(igci)),
    ('IGCI', ['B', 'B type', 'A', 'A type'], MultiColumnTransform(igci)),
    ('Sub', ['IGCI[A,A type,B,B type]', 'IGCI[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[IGCI[A,A type,B,B type],IGCI[B,B type,A,A type]]', SimpleTransform(abs)),

    ('Uniform Divergence', ['A', 'A type'], MultiColumnTransform(uniform_divergence)),
    ('Uniform Divergence', ['B', 'B type'], MultiColumnTransform(uniform_divergence)),
    ('Max', ['Uniform Divergence[A,A type]', 'Uniform Divergence[B,B type]'], MultiColumnTransform(max)),
    ('Min', ['Uniform Divergence[A,A type]', 'Uniform Divergence[B,B type]'], MultiColumnTransform(min)),
    ('Sub', ['Uniform Divergence[A,A type]', 'Uniform Divergence[B,B type]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Uniform Divergence[A,A type],Uniform Divergence[B,B type]]', SimpleTransform(abs)),

    ('Discrete Entropy', ['A', 'A type'], MultiColumnTransform(discrete_entropy)),
    ('Discrete Entropy', ['B', 'B type'], MultiColumnTransform(discrete_entropy)),
    ('Max', ['Discrete Entropy[A,A type]', 'Discrete Entropy[B,B type]'], MultiColumnTransform(max)),
    ('Min', ['Discrete Entropy[A,A type]', 'Discrete Entropy[B,B type]'], MultiColumnTransform(min)),
    ('Sub', ['Discrete Entropy[A,A type]', 'Discrete Entropy[B,B type]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]', SimpleTransform(abs)),

    ('Normalized Discrete Entropy', ['A', 'A type'], MultiColumnTransform(normalized_discrete_entropy)),
    ('Normalized Discrete Entropy', ['B', 'B type'], MultiColumnTransform(normalized_discrete_entropy)),
    ('Max', ['Normalized Discrete Entropy[A,A type]', 'Normalized Discrete Entropy[B,B type]'],
     MultiColumnTransform(max)),
    ('Min', ['Normalized Discrete Entropy[A,A type]', 'Normalized Discrete Entropy[B,B type]'],
     MultiColumnTransform(min)),
    ('Sub', ['Normalized Discrete Entropy[A,A type]', 'Normalized Discrete Entropy[B,B type]'],
     MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Normalized Discrete Entropy[A,A type],Normalized Discrete Entropy[B,B type]]', SimpleTransform(abs)),

    ('Discrete Joint Entropy', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(discrete_joint_entropy)),
    ('Normalized Discrete Joint Entropy', ['A', 'A type', 'B', 'B type'],
     MultiColumnTransform(normalized_discrete_joint_entropy)),
    (
    'Discrete Conditional Entropy', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(discrete_conditional_entropy)),
    (
    'Discrete Conditional Entropy', ['B', 'B type', 'A', 'A type'], MultiColumnTransform(discrete_conditional_entropy)),
    ('Discrete Mutual Information', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(discrete_mutual_information)),
    ('Normalized Discrete Mutual Information',
     ['Discrete Mutual Information[A,A type,B,B type]', 'Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]'],
     MultiColumnTransform(operator.truediv)),
    ('Normalized Discrete Mutual Information',
     ['Discrete Mutual Information[A,A type,B,B type]', 'Discrete Joint Entropy[A,A type,B,B type]'],
     MultiColumnTransform(operator.truediv)),
    ('Adjusted Mutual Information', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(adjusted_mutual_information)),

    ('Polyfit', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(fit)),
    ('Polyfit', ['B', 'B type', 'A', 'A type'], MultiColumnTransform(fit)),
    ('Sub', ['Polyfit[A,A type,B,B type]', 'Polyfit[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Polyfit[A,A type,B,B type],Polyfit[B,B type,A,A type]]', SimpleTransform(abs)),

    ('Polyfit Error', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(fit_error)),
    ('Polyfit Error', ['B', 'B type', 'A', 'A type'], MultiColumnTransform(fit_error)),
    ('Sub', ['Polyfit Error[A,A type,B,B type]', 'Polyfit Error[B,B type,A,A type]'],
     MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Polyfit Error[A,A type,B,B type],Polyfit Error[B,B type,A,A type]]', SimpleTransform(abs)),

    (
    'Normalized Error Probability', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(normalized_error_probability)),
    (
    'Normalized Error Probability', ['B', 'B type', 'A', 'A type'], MultiColumnTransform(normalized_error_probability)),
    ('Sub', ['Normalized Error Probability[A,A type,B,B type]', 'Normalized Error Probability[B,B type,A,A type]'],
     MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Normalized Error Probability[A,A type,B,B type],Normalized Error Probability[B,B type,A,A type]]',
     SimpleTransform(abs)),

    ('Conditional Distribution Entropy Variance', ['A', 'A type', 'B', 'B type'],
     MultiColumnTransform(fit_noise_entropy)),
    ('Conditional Distribution Entropy Variance', ['B', 'B type', 'A', 'A type'],
     MultiColumnTransform(fit_noise_entropy)),
    ('Sub', ['Conditional Distribution Entropy Variance[A,A type,B,B type]',
             'Conditional Distribution Entropy Variance[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
    ('Abs',
     'Sub[Conditional Distribution Entropy Variance[A,A type,B,B type],Conditional Distribution Entropy Variance[B,B type,A,A type]]',
     SimpleTransform(abs)),

    ('Conditional Distribution Skewness Variance', ['A', 'A type', 'B', 'B type'],
     MultiColumnTransform(fit_noise_skewness)),
    ('Conditional Distribution Skewness Variance', ['B', 'B type', 'A', 'A type'],
     MultiColumnTransform(fit_noise_skewness)),
    ('Sub', ['Conditional Distribution Skewness Variance[A,A type,B,B type]',
             'Conditional Distribution Skewness Variance[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
    ('Abs',
     'Sub[Conditional Distribution Skewness Variance[A,A type,B,B type],Conditional Distribution Skewness Variance[B,B type,A,A type]]',
     SimpleTransform(abs)),

    ('Conditional Distribution Kurtosis Variance', ['A', 'A type', 'B', 'B type'],
     MultiColumnTransform(fit_noise_kurtosis)),
    ('Conditional Distribution Kurtosis Variance', ['B', 'B type', 'A', 'A type'],
     MultiColumnTransform(fit_noise_kurtosis)),
    ('Sub', ['Conditional Distribution Kurtosis Variance[A,A type,B,B type]',
             'Conditional Distribution Kurtosis Variance[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
    ('Abs',
     'Sub[Conditional Distribution Kurtosis Variance[A,A type,B,B type],Conditional Distribution Kurtosis Variance[B,B type,A,A type]]',
     SimpleTransform(abs)),

    ('Conditional Distribution Similarity', ['A', 'A type', 'B', 'B type'],
     MultiColumnTransform(conditional_distribution_similarity)),
    ('Conditional Distribution Similarity', ['B', 'B type', 'A', 'A type'],
     MultiColumnTransform(conditional_distribution_similarity)),
    ('Sub', ['Conditional Distribution Similarity[A,A type,B,B type]',
             'Conditional Distribution Similarity[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
    ('Abs',
     'Sub[Conditional Distribution Similarity[A,A type,B,B type],Conditional Distribution Similarity[B,B type,A,A type]]',
     SimpleTransform(abs)),

    ('Moment21', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(moment21)),
    ('Moment21', ['B', 'B type', 'A', 'A type'], MultiColumnTransform(moment21)),
    ('Sub', ['Moment21[A,A type,B,B type]', 'Moment21[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Moment21[A,A type,B,B type],Moment21[B,B type,A,A type]]', SimpleTransform(abs)),

    ('Abs', 'Moment21[A,A type,B,B type]', SimpleTransform(abs)),
    ('Abs', 'Moment21[B,B type,A,A type]', SimpleTransform(abs)),
    ('Sub', ['Abs[Moment21[A,A type,B,B type]]', 'Abs[Moment21[B,B type,A,A type]]'],
     MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Abs[Moment21[A,A type,B,B type]],Abs[Moment21[B,B type,A,A type]]]', SimpleTransform(abs)),

    ('Moment31', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(moment31)),
    ('Moment31', ['B', 'B type', 'A', 'A type'], MultiColumnTransform(moment31)),
    ('Sub', ['Moment31[A,A type,B,B type]', 'Moment31[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Moment31[A,A type,B,B type],Moment31[B,B type,A,A type]]', SimpleTransform(abs)),

    ('Abs', 'Moment31[A,A type,B,B type]', SimpleTransform(abs)),
    ('Abs', 'Moment31[B,B type,A,A type]', SimpleTransform(abs)),
    ('Sub', ['Abs[Moment31[A,A type,B,B type]]', 'Abs[Moment31[B,B type,A,A type]]'],
     MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Abs[Moment31[A,A type,B,B type]],Abs[Moment31[B,B type,A,A type]]]', SimpleTransform(abs)),

    ('Skewness', ['A', 'A type'], MultiColumnTransform(normalized_skewness)),
    ('Skewness', ['B', 'B type'], MultiColumnTransform(normalized_skewness)),
    ('Sub', ['Skewness[A,A type]', 'Skewness[B,B type]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Skewness[A,A type],Skewness[B,B type]]', SimpleTransform(abs)),

    ('Abs', 'Skewness[A,A type]', SimpleTransform(abs)),
    ('Abs', 'Skewness[B,B type]', SimpleTransform(abs)),
    ('Max', ['Abs[Skewness[A,A type]]', 'Abs[Skewness[B,B type]]'], MultiColumnTransform(max)),
    ('Min', ['Abs[Skewness[A,A type]]', 'Abs[Skewness[B,B type]]'], MultiColumnTransform(min)),
    ('Sub', ['Abs[Skewness[A,A type]]', 'Abs[Skewness[B,B type]]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Abs[Skewness[A,A type]],Abs[Skewness[B,B type]]]', SimpleTransform(abs)),

    ('Kurtosis', ['A', 'A type'], MultiColumnTransform(normalized_kurtosis)),
    ('Kurtosis', ['B', 'B type'], MultiColumnTransform(normalized_kurtosis)),
    ('Max', ['Kurtosis[A,A type]', 'Kurtosis[B,B type]'], MultiColumnTransform(max)),
    ('Min', ['Kurtosis[A,A type]', 'Kurtosis[B,B type]'], MultiColumnTransform(min)),
    ('Sub', ['Kurtosis[A,A type]', 'Kurtosis[B,B type]'], MultiColumnTransform(operator.sub)),
    ('Abs', 'Sub[Kurtosis[A,A type],Kurtosis[B,B type]]', SimpleTransform(abs)),

    ('HSIC', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(normalized_hsic)),
    ('Pearson R', ['A', 'A type', 'B', 'B type'], MultiColumnTransform(correlation)),
    ('Abs', 'Pearson R[A,A type,B,B type]', SimpleTransform(abs))
]


def calculate_method(args):
    obj = args[0]
    name = args[1]
    margs = args[2]
    method = getattr(obj, name)
    return method(*margs)


def extract_features(X, features=all_features, y=None, njobs=-1):
    if njobs != 1:
        pool = Pool(njobs if njobs != -1 else None)
        pmap = pool.map
    else:
        pmap = map

    def complete_feature_name(feature_name, column_names):
        if type(column_names) is list:
            long_feature_name = feature_name + '[' + ','.join(column_names) + ']'
        else:
            long_feature_name = feature_name + '[' + column_names + ']'
        if feature_name[0] == '+':
            long_feature_name = long_feature_name[1:]
        return long_feature_name

    def is_in_X(column_names):
        if type(column_names) is list:
            return set(column_names).issubset(X.columns)
        else:
            return column_names in X.columns

    def can_be_extracted(feature_name, column_names):
        long_feature_name = complete_feature_name(feature_name, column_names)
        to_be_extracted = ((feature_name[0] == '+') or (long_feature_name not in X.columns))

        # print(long_feature_name, to_be_extracted and is_in_X(column_names))
        return to_be_extracted and is_in_X(column_names)

    while True:
        for typefeature, var in [("A type","A"), ("B type", "B")]:
            if typefeature not in X.columns:
                X[typefeature] = determine_type(X[var])
        new_features_list = [(complete_feature_name(feature_name, column_names), column_names, extractor)
                             for feature_name, column_names, extractor in features if
                             can_be_extracted(feature_name, column_names)]
        if not new_features_list:
            break
        # print(new_features_list)
        task = [(extractor, 'fit_transform', (X[column_names], y)) for _, column_names, extractor in new_features_list]
        new_features = pmap(calculate_method, task)
        for (feature_name, _, _), feature in zip(new_features_list, new_features):
            try:
                X[feature_name] = feature
            except ValueError:
                X[feature_name] = feature.transpose()
        #print(X.columns)
    return X
