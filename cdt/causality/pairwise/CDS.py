"""
Conditional Distribution Similarity Statistic
Used to infer causal directions
Author : José A.R. Fonollosa
Ref : Fonollosa, José AR, "Conditional distribution variability measures for causality detection", 2016.

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

import numpy as np
from collections import Counter
from .model import PairwiseModel
import pandas as pd

BINARY = "Binary"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"


def count_unique(x):
    try:
        if type(x) == np.ndarray:
            return len(np.unique(x))
        else:
            return len(set(x))
    except TypeError as e:
        print(x)
        raise e


def numerical(tp):
    assert type(tp) is str
    return tp == NUMERICAL


def len_discretized_values(x, tx, ffactor, maxdev):
    return len(discretized_values(x, tx, ffactor, maxdev))


def discretized_values(x, tx, ffactor, maxdev):
    if numerical(tx) and count_unique(x) > (2 * ffactor * maxdev + 1):
        vmax = ffactor * maxdev
        vmin = -ffactor * maxdev
        return range(vmin, vmax + 1)
    else:
        return sorted(list(set(x)))


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


def discretized_sequences(x, y, ffactor=3, maxdev=3):
    return discretized_sequence(x, "Numerical", ffactor, maxdev), discretized_sequence(y, "Numerical", ffactor,
                                                                                       maxdev)


class CDS(PairwiseModel):
    """Conditional Distribution Similarity Statistic

    **Description:** The Conditional Distribution Similarity Statistic measures the
    std. of the rescaled values of y (resp. x) after binning in the x (resp. y) direction.
    The lower the std. the more likely the pair to be x->y (resp. y->x). It is
    a single feature of the Jarfo model.

    **Data Type**: Continuous and Discrete

    **Assumptions**: This approach is a statistical feature of the
    joint distribution of the data mesuring the variance of the marginals, after
    conditioning on bins.

    .. note::
       Ref : Fonollosa, José AR, "Conditional distribution variability measures for causality detection", 2016.

    Example:
        >>> from cdt.causality.pairwise import CDS
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from cdt.data import load_dataset
        >>> data, labels = load_dataset('tuebingen')
        >>> obj = CDS()
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
        super(CDS, self).__init__()
        self.ffactor = ffactor
        self.maxdev = maxdev
        self.minc = minc

    def predict_proba(self, dataset, **kwargs):
        """ Infer causal relationships between 2 variables using the CDS statistic

        Args:
            dataset (tuple): Couple of np.ndarray variables to classify

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        a, b = dataset
        return self.cds_score(b, a) - self.cds_score(a, b)

    def cds_score(self, x_te, y_te):
        """ Computes the cds statistic from variable 1 to variable 2

        Args:
            x_te (numpy.ndarray): Variable 1
            y_te (numpy.ndarray): Variable 2

        Returns:
            float: CDS fit score
        """
        if type(x_te) == np.ndarray:
            x_te, y_te = pd.Series(x_te.reshape(-1)), pd.Series(y_te.reshape(-1))
        xd, yd = discretized_sequences(x_te,  y_te,  self.ffactor, self.maxdev)
        cx = Counter(xd)
        cy = Counter(yd)
        yrange = sorted(cy.keys())
        ny = len(yrange)
        py = np.array([cy[i] for i in yrange], dtype=float)
        py = py / py.sum()
        pyx = []
        for a in cx:
            if cx[a] > self.minc:
                yx = y_te[xd == a]
                # if not numerical(ty):
                #     cyx = Counter(yx)
                #     pyxa = np.array([cyx[i] for i in yrange], dtype=float)
                #     pyxa.sort()
                if count_unique(y_te) > len_discretized_values(y_te, "Numerical", self.ffactor, self.maxdev):

                    yx = (yx - np.mean(yx)) / np.std(y_te)
                    yx = discretized_sequence(yx, "Numerical", self.ffactor, self.maxdev, norm=False)
                    cyx = Counter(yx.astype(int))
                    pyxa = np.array([cyx[i] for i in discretized_values(y_te, "Numerical", self.ffactor, self.maxdev)],
                                    dtype=float)

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

        if len(pyx) == 0:
            return 0

        pyx = np.array(pyx)
        pyx = pyx - pyx.mean(axis=0)
        return np.std(pyx)
