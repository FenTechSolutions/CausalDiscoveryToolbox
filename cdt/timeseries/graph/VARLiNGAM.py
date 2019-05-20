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
import pandas as pd
import networkx as nx
from statsmodels.tsa.vector_ar.var_model import VAR
from ...causality.graph import LiNGAM
from ...causality.graph.model import GraphModel
from ...utils.Settings import SETTINGS


class VarLiNGAM(GraphModel):
    """ Estimate a VAR-LiNGAM
    Random generate matrix ids to set zeros.

    Args:
        lag (float): order to estimate the vector autoregressive model
        verbose (bool): Verbosity of the class. Defaults to SETTINGS.verbose

    .. note::
       Ref: - A. Hyvarinen, S. Shimizu, P.O. Hoyer ((ICML-2008). Causal modelling
       combining instantaneous and lagged effects: an identifiable model based
       on non-Gaussianity;
       - A. Hyvarinen, K. Zhang, S. Shimizu, P.O. Hoyer (JMLR-2010). Estimation of
       a Structural Vector Autoregression Model Using Non-Gaussianity;
     """

    def __init__(self, lag=1, verbose=None):
        self.lag = lag
        self.verbose = SETTINGS.get_default(verbose=verbose)

    def orient_undirected_graph(self, data, graph):
        """Run varLiNGAM on an undirected graph."""
        # Building setup w/ arguments.
        raise ValueError("VarLiNGAM cannot (yet) be ran with a skeleton/directed graph.")

    def orient_directed_graph(self, data, graph):
        """Run varLiNGAM on a directed_graph."""
        raise ValueError("VarLiNGAM cannot (yet) be ran with a skeleton/directed graph.")

    def create_graph_from_data(self, data):
        """ Run the VarLiNGAM algorithm on data.

        Args:
            data (pandas.DataFrame): time series data

        Returns:
            tuple :(networkx.Digraph, networkx.Digraph) Predictions given by
               the varLiNGAM algorithm: Instantaneous and Lagged causal Graphs
        """
        inst, lagged = self._run_varLiNGAM(data.values, verbose=self.verbose)
        return (nx.relabel_nodes(nx.DiGraph(inst),
                                 {idx: i for idx, i in enumerate(data.columns)}),
                nx.relabel_nodes(nx.DiGraph(lagged),
                                 {idx: i for idx, i in enumerate(data.columns)}),
                )

    def _run_varLiNGAM(self, xt, verbose=False):
        """ Run the VarLiNGAM algorithm on data.

        Args:
            xt : time series matrix with size n*m (length*num_variables)

        Returns:
            Tuple: (Bo, Bhat) Instantaneous and lagged causal coefficients

        """
        Ident = np.identity(xt.shape[1])

        # Step 1: VAR estimation
        model = VAR(xt)
        results = model.fit(self.lag)
        Mt_ = results.params[1:, :]

        # Step 2: LiNGAM on Residuals
        resid_VAR = results.resid
        model = LiNGAM(verbose=verbose)
        data = pd.DataFrame(resid_VAR)
        Bo_ = model._run_LiNGAM(data)

        # Step 3: Get instantaneous matrix Bo from LiNGAM
        # Bo_ = pd.read_csv("results.csv").values

        # Step 4: Calculation of lagged Bhat
        Bhat_ = np.dot((Ident - Bo_), Mt_)
        return (Bo_, Bhat_)
