"""Graph causal models base class.

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


class GraphModel(object):
    """Base class for all graph causal inference models.

    Usage for undirected/directed graphs and raw data. All causal discovery
    models out of observational data base themselves on this class. Its main
    feature is the predict function that executes a function according to the
    given arguments.
    """

    def __init__(self):
        """Init."""
        super(GraphModel, self).__init__()

    def predict(self, df_data, graph=None, **kwargs):
        """Orient a graph using the method defined by the arguments.

        Depending on the type of `graph`, this function process to execute
        different functions:

        1. If ``graph`` is a ``networkx.DiGraph``, then ``self.orient_directed_graph`` is executed.
        2. If ``graph`` is a ``networkx.Graph``, then ``self.orient_undirected_graph`` is executed.
        3. If ``graph`` is a ``None``, then ``self.create_graph_from_data`` is executed.

        Args:
            df_data (pandas.DataFrame): DataFrame containing the observational data.
            graph (networkx.DiGraph or networkx.Graph or None): Prior knowledge on the causal graph.

        .. warning::
           Requirement : Name of the nodes in the graph must correspond to the
           name of the variables in df_data
        """
        if graph is None:
            return self.create_graph_from_data(df_data, **kwargs)
        elif isinstance(graph, nx.DiGraph):
            return self.orient_directed_graph(df_data, graph, **kwargs)
        elif isinstance(graph, nx.Graph):
            return self.orient_undirected_graph(df_data, graph, **kwargs)
        else:
            print('Unknown Graph type')
            raise ValueError

    def orient_undirected_graph(self, data, umg, **kwargs):
        """Orient an undirected graph.

        .. note::
           Not implemented: will be implemented by the model classes.
        """
        raise NotImplementedError

    def orient_directed_graph(self, data, dag, **kwargs):
        """Re/Orient an undirected graph.

        .. note::
           Not implemented: will be implemented by the model classes.
        """
        raise NotImplementedError

    def create_graph_from_data(self, data, **kwargs):
        """Infer a directed graph out of data.

        .. note::
           Not implemented: will be implemented by the model classes.
        """
        raise NotImplementedError

from .SAM import SAM_generators, SAM_discriminator
from .CGNN import CGNN_model
