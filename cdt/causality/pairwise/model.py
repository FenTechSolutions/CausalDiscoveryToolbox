"""
Pairwise causal models base class
Author: Diviyan Kalainathan
Date : 7/06/2017

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
import networkx as nx
from sklearn.preprocessing import scale
from pandas import DataFrame, Series
from ...utils.Settings import SETTINGS


class PairwiseModel(object):
    """Base class for all pairwise causal inference models

    Usage for undirected/directed graphs and CEPC df format.
    """

    def __init__(self):
        """Init."""
        super(PairwiseModel, self).__init__()

    def predict(self, x, *args, **kwargs):
        """Generic predict method, chooses which subfunction to use for a more
        suited.

        Depending on the type of `x` and of `*args`, this function process to execute
        different functions in the priority order:

        1. If ``args[0]`` is a ``networkx.(Di)Graph``, then ``self.orient_graph`` is executed.
        2. If ``args[0]`` exists, then ``self.predict_proba`` is executed.
        3. If ``x`` is a ``pandas.DataFrame``, then ``self.predict_dataset`` is executed.
        4. If ``x`` is a ``pandas.Series``, then ``self.predict_proba`` is executed.

        Args:
            x (numpy.array or pandas.DataFrame or pandas.Series): First variable or dataset.
            args (numpy.array or networkx.Graph): graph or second variable.

        Returns:
            pandas.Dataframe or networkx.Digraph: predictions output
        """
        if len(args) > 0:
            if type(args[0]) == nx.Graph or type(args[0]) == nx.DiGraph:
                return self.orient_graph(x, *args, **kwargs)
            else:
                y = args.pop(0)
                return self.predict_proba((x, y), *args, **kwargs)
        elif type(x) == DataFrame:
            return self.predict_dataset(x, *args, **kwargs)
        elif type(x) == Series:
            return self.predict_proba((x.iloc[0], x.iloc[1]), *args, **kwargs)

    def predict_proba(self, dataset, idx=0, **kwargs):
        """Prediction method for pairwise causal inference.

        predict_proba is meant to be overridden in all subclasses

        Args:
            dataset (tuple): Couple of np.ndarray variables to classify
            idx (int): (optional) index number for printing purposes

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        raise NotImplementedError

    def predict_dataset(self, x, **kwargs):
        """Generic dataset prediction function.

        Runs the score independently on all pairs.

        Args:
            x (pandas.DataFrame): a CEPC format Dataframe.
            kwargs (dict): additional arguments for the algorithms

        Returns:
            pandas.DataFrame: a Dataframe with the predictions.
        """
        printout = kwargs.get("printout", None)
        pred = []
        res = []
        x.columns = ["A", "B"]
        for idx, row in x.iterrows():
            a = scale(row['A'].reshape((len(row['A']), 1)))
            b = scale(row['B'].reshape((len(row['B']), 1)))

            pred.append(self.predict_proba((a, b), idx=idx))

            if printout is not None:
                res.append([row['SampleID'], pred[-1]])
                DataFrame(res, columns=['SampleID', 'Predictions']).to_csv(
                    printout, index=False)
        return pred

    def orient_graph(self, df_data, graph, printout=None, **kwargs):
        """Orient an undirected graph using the pairwise method defined by the subclass.

        The pairwise method is ran on every undirected edge.

        Args:
            df_data (pandas.DataFrame): Data
            graph (networkx.Graph): Graph to orient
            printout (str): (optional) Path to file where to save temporary results

        Returns:
            networkx.DiGraph: a directed graph, which might contain cycles

        .. warning::
           Requirement : Name of the nodes in the graph correspond to name of
           the variables in df_data
        """
        if isinstance(graph, nx.DiGraph):
            edges = [a for a in list(graph.edges()) if (a[1], a[0]) in list(graph.edges())]
            oriented_edges = [a for a in list(graph.edges()) if (a[1], a[0]) not in list(graph.edges())]
            for a in edges:
                if (a[1], a[0]) in list(graph.edges()):
                    edges.remove(a)
            output = nx.DiGraph()
            for i in oriented_edges:
                output.add_edge(*i)

        elif isinstance(graph, nx.Graph):
            edges = list(graph.edges())
            output = nx.DiGraph()

        else:
            raise TypeError("Data type not understood.")

        res = []

        for idx, (a, b) in enumerate(edges):
            weight = self.predict_proba(
                (df_data[a].values.reshape((-1, 1)),
                 df_data[b].values.reshape((-1, 1))), idx=idx,
                **kwargs)
            if weight > 0:  # a causes b
                output.add_edge(a, b, weight=weight)
            elif weight < 0:
                output.add_edge(b, a, weight=abs(weight))
            if printout is not None:
                res.append([str(a) + '-' + str(b), weight])
                DataFrame(res, columns=['SampleID', 'Predictions']).to_csv(
                    printout, index=False)

        for node in list(df_data.columns.values):
            if node not in output.nodes():
                output.add_node(node)

        return output

from .GNN import GNN_model
from .NCC import NCC_model
