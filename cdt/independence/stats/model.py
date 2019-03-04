"""Base class for dependence models.

Author: Diviyan Kalainathan

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
from networkx import Graph


class IndependenceModel(object):
    """Base class for independence and utilities to recover the
    undirected graph out of data.

    Args:
        predictor (function): function to estimate dependence (0 : independence), taking as input 2 array-like variables.

    """

    def __init__(self, predictor=None):
        """Init the model."""
        super(IndependenceModel, self).__init__()
        if predictor is not None:
            self.predict = predictor

    def predict(self, a, b):
        """Compute a dependence test statistic between variables.

        Args:
            a (numpy.ndarray): First variable
            b (numpy.ndarray): Second variable

        Returns:
            float: dependence test statistic (close to 0 -> independent)
        """
        raise NotImplementedError

    def predict_undirected_graph(self, data):
        """Build a skeleton using a pairwise independence criterion.

        Args:
            data (pandas.DataFrame): Raw data table

        Returns:
            networkx.Graph: Undirected graph representing the skeleton.
        """
        graph = Graph()

        for idx_i, i in enumerate(data.columns):
            for idx_j, j in enumerate(data.columns[idx_i+1:]):
                score = self.predict(data[i].values, data[j].values)
                if abs(score) > 0.001:
                    graph.add_edge(i, j, weight=score)

        return graph
