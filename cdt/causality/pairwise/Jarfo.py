"""
Jarfo causal inference model
Author : José AR Fonollosa
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

from pandas import DataFrame
import networkx as nx
from .Jarfo_model import train
from .model import PairwiseModel
from .Jarfo_model import predict
from copy import deepcopy


class Jarfo(PairwiseModel):
    """Jarfo model, 2nd of the Cause Effect Pairs challenge,
    1st of the Fast Causation Challenge.
    Builds lots of features on top of a gradient boosting classifier.

    .. note::
       Ref : Fonollosa, José AR, "Conditional distribution variability measures for causality detection", 2016.
    """
    def __init__(self):
        super(Jarfo, self).__init__()

    def fit(self, df, tar):
        df2 = DataFrame()
        tar2 = DataFrame()
        for idx, row in df.iterrows():
            df2 = df2.append(row, ignore_index=True)
            df2 = df2.append({'A': row["B"], 'B': row["A"]}, ignore_index=True)
        for idx, row in tar.iterrows():
            tar2 = tar2.append(row, ignore_index=True)
            tar2 = tar2.append(-row, ignore_index=True)
        self.model = train.train(df2, tar2)

    def predict_dataset(self, df):
        """Runs Jarfo independently on all pairs.

        Args:
            x (pandas.DataFrame): a CEPC format Dataframe.
            kwargs (dict): additional arguments for the algorithms

        Returns:
            pandas.DataFrame: a Dataframe with the predictions.
        """
        if len(list(df.columns)) == 2:
            df.columns = ["A", "B"]
        if self.model is None:
            raise AssertionError("Model has not been trained before predictions")
        df2 = DataFrame()

        for idx, row in df.iterrows():
            df2 = df2.append(row, ignore_index=True)
            df2 = df2.append({'A': row["B"], 'B': row["A"]}, ignore_index=True)
        return predict.predict(deepcopy(df2), deepcopy(self.model))[::2]

    def predict_proba(self, a, b, idx=0, **kwargs):
        """ Use Jarfo to predict the causal direction of a pair of vars.

        Args:
            a (numpy.ndarray): Variable 1
            b (numpy.ndarray): Variable 2
            idx (int): (optional) index number for printing purposes

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        return self.predict_dataset(DataFrame([[a, b]],
                                              columns=['A', 'B']))

    def orient_graph(self, df_data, graph, printout=None, **kwargs):
        """Orient an undirected graph using Jarfo, function modified for optimization.

        Args:
            df_data (pandas.DataFrame): Data
            umg (networkx.Graph): Graph to orient
            nb_runs (int): number of times to rerun for each pair (bootstrap)
            printout (str): (optional) Path to file where to save temporary results

        Returns:
            networkx.DiGraph: a directed graph, which might contain cycles

        .. warning:
           Requirement : Name of the nodes in the graph correspond to name of
           the variables in df_data

        """
        if type(graph) == nx.DiGraph:
            edges = [a for a in list(graph.edges()) if (a[1], a[0]) in list(graph.edges())]
            oriented_edges = [a for a in list(graph.edges()) if (a[1], a[0]) not in list(graph.edges())]
            for a in edges:
                if (a[1], a[0]) in list(graph.edges()):
                    edges.remove(a)
            output = nx.DiGraph()
            for i in oriented_edges:
                output.add_edge(*i)

        elif type(graph) == nx.Graph:
            edges = list(graph.edges())
            output = nx.DiGraph()

        else:
            raise TypeError("Data type not understood.")

        res = []
        df_task = DataFrame()
        for idx, (a, b) in enumerate(edges):
            df_task = df_task.append({'A': df_data[a].values.reshape((-1, 1)),
                                      'B': df_data[b].values.reshape((-1, 1))},
                                     ignore_index=True)
        weights = self.predict_dataset(df_task)
        for weight, (a, b) in zip(weights, edges):
            if weight > 0:  # a causes b
                output.add_edge(a, b, weight=weight)
            else:
                output.add_edge(b, a, weight=abs(weight))
            if printout is not None:
                res.append([str(a) + '-' + str(b), weight])
                DataFrame(res, columns=['SampleID', 'Predictions']).to_csv(
                    printout, index=False)

        for node in list(df_data.columns.values):
            if node not in output.nodes():
                output.add_node(node)

        return output
