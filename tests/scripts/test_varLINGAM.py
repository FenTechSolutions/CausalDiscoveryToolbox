"""VARLiNGAM algorithm.

Author: Georgios Koutroulis

.. MIT License
..
.. Copyright (c) 2019 Georgios Koutroulis
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
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from cdt.timeseries.graph import VarLiNGAM
import networkx as nx
import pandas as pd


def setRCmatrix(m, p):
    """Random generate matrix ids to set zeros.
    Parameters
    ----------
    m : integer
        Number of rows/columns of square matrix
    p : float (0. < p < 1.0)
        percent of entries in matrix that set to zero.
    Returns
    -------
    rowsZeros, columnsZeros : Tuple
        Rows and columns id of matrix
        that will be set to zero
    """
    idZeros = np.random.choice(np.arange(m*m), int(p*m*m), replace=False)
    rowsZeros = []
    columnsZeros = []
    for i in idZeros:
        if i > m-1:
            rowsZeros.append(i//m)
            columnsZeros.append(i%m)
        else:
            rowsZeros.append(i)
            columnsZeros.append(1)
    return (rowsZeros, columnsZeros)


def test_VARLiNGAM(m=5, n=300, plot=False):
    # Test VARLiNGAM on simulated data reproduced from the paper
    # - A. HyvÃ¤rinen, K. Zhang, S. Shimizu, P.O. Hoyer (JMLR-2010). Estimation of
    #   a Structural Vector Autoregression Model Using Non-Gaussianity;

    # Randomly construct lower triangular instantaneuous causality matrix Bo
    Bo = np.random.uniform(0.05, 0.5, size=(m,m))
    # Bo = np.resize(Bo,(m, m))
    Bo = np.tril(Bo)
    Bo[np.arange(m), np.arange(m)] = 0

    signs = np.random.choice([-1, 1],size=m*m)
    signs = np.resize(signs, (m, m))
    Bo = Bo*signs

    # Randomly construct lower triangular lagged causality matrix B1 (60% set to zero)
    B1 = np.random.uniform(0.05, 0.5, size=(m,m))
    signs = np.random.choice([-1, 1],size=m*m)
    signs = np.resize(signs, (m, m))
    B1 = B1*signs

    # Set 60% of the entries to zero
    rowsZeros, columnsZeros = setRCmatrix(m, 0.6)
    B1[rowsZeros, columnsZeros] = .0

    rowsZeros, columnsZeros = setRCmatrix(m, 0.6)
    Bo[rowsZeros, columnsZeros] = .0

    #Generate non-Gaussian disturbances
    q = np.random.uniform(m) * 0.5 + 1.5

    e = np.random.randn(n, m)
    e = np.sign(e) * (np.abs(e)**q)

    #Generate time series with predefined length
    xt = np.zeros((n, m))
    I = np.identity(m)
    M1 = np.dot(np.linalg.inv(I - Bo), B1)
    Bo_t = []
    rowsZeros, colsZeros = np.where(Bo != 0.)

    # Generate time series according to autoregressive and causal Matrices
    for i in range(n):
        xt[i, :] = np.dot(xt[i-1, :], M1) + np.dot(np.linalg.inv(I - Bo), e[i, :])

    #Test VARLiNGAM
    model = VarLiNGAM()
    Bo_, Bhat_ = model._run_varLiNGAM(xt)

    if plot:
        fig = plt.figure(figsize=(12,4))
        fig.subplots_adjust(hspace=0.1, wspace=0.3)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_xlim((-1, 1))
        ax1.set_ylim((-1, 1))
        ax1.set_xlabel("Generating Bo")
        ax1.set_ylabel("Estimated Bo")
        ax2.set_xlim((-1, 1))
        ax2.set_ylim((-1, 1))
        ax2.set_xlabel("Generating Bhat")
        ax2.set_ylabel("Estimated Bhat")
        ax1.plot((-1, 1), (-1, 1), "r--")
        ax2.plot((-1, 1), (-1, 1), "r--")

        idR, idC = np.where(Bo != 0)
        for ij in list(zip(idR, idC)):
            ax1.scatter(Bo[ij], Bo_[ij], c="b")
            ax2.scatter(B1[ij], Bhat_[ij], c="b")
        plt.show()
    else:
        return (Bo_, Bhat_)


def test_pipeline():
    data = pd.DataFrame(np.random.randn(500, 10))
    model = VarLiNGAM()
    out1, out2 = model.predict(data)
    assert type(out1) == nx.DiGraph
    assert type(out1) == nx.DiGraph


if __name__ == '__main__':
    test_VARLiNGAM(10, 500)
    test_pipeline()
