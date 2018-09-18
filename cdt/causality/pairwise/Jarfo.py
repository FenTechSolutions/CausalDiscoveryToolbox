"""
Jarfo causal inference model
Author : José AR Fonollosa
Ref : Fonollosa, José AR, "Conditional distribution variability measures for causality detection", 2016.
"""

from pandas import DataFrame
import networkx as nx
from .Jarfo_model import train
from .model import PairwiseModel
from .Jarfo_model import predict
from copy import deepcopy


class Jarfo(PairwiseModel):
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
        if len(list(df.columns)) == 2:
            df.columns = ["A", "B"]
        if self.model is None:
            raise AssertionError("Model has not been trained before predictions")
        df2 = DataFrame()

        for idx, row in df.iterrows():
            df2 = df2.append(row, ignore_index=True)
            df2 = df2.append({'A': row["B"], 'B': row["A"]}, ignore_index=True)
        return predict.predict(deepcopy(df2), deepcopy(self.model))[::2]

    def predict_proba(self, a, b, idx, **kwargs):
        return self.predict_dataset(DataFrame([[a, b]],
                                              columns=['A', 'B']))

    def orient_graph(self, df_data, graph, printout=None, nb_runs=6, **kwargs):
        """Orient an undirected graph using the pairwise method defined by the subclass.

        Requirement : Name of the nodes in the graph correspond to name of the variables in df_data
        :param df_data: dataset
        :param umg: UndirectedGraph
        :param printout: print regularly predictions
        :return: Directed graph w/ weights
        :rtype: DirectedGraph
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
