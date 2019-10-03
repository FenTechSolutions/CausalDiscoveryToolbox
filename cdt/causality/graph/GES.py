"""GES algorithm.

Imported from the Pcalg package.
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
import os
import uuid
import warnings
import networkx as nx
from shutil import rmtree
from .model import GraphModel
from pandas import DataFrame, read_csv
from ...utils.R import RPackages, launch_R_script
from ...utils.Settings import SETTINGS


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class GES(GraphModel):
    """GES algorithm **[R model]**.

    **Description:** Greedy Equivalence Search algorithm. A score-based
    Bayesian algorithm that searches heuristically the graph which minimizes
    a likelihood score on the data.

    **Required R packages**: pcalg

    **Data Type:** Continuous (``score='obs'``) or Categorical (``score='int'``)

    **Assumptions:** The output is a Partially Directed Acyclic Graph (PDAG)
    (A markov equivalence class). The available scores assume linearity of
    mechanisms and gaussianity of the data.

    Args:
        score (str): Sets the score used by GES.
        verbose (bool): Defaults to ``cdt.SETTINGS.verbose``.

    Available scores:
        + int: GaussL0penIntScore
        + obs: GaussL0penObsScore

    .. note::
       Ref:
       D.M. Chickering (2002).  Optimal structure identification with greedy search.
       Journal of Machine Learning Research 3 , 507–554

       A. Hauser and P. Bühlmann (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs.
       Journal of Machine Learning Research 13, 2409–2464.

       P. Nandy, A. Hauser and M. Maathuis (2015). Understanding consistency in
       hybrid causal structure learning.
       arXiv preprint 1507.02608

       P. Spirtes, C.N. Glymour, and R. Scheines (2000).
       Causation, Prediction, and Search, MIT Press, Cambridge (MA)

    Example:
        >>> import networkx as nx
        >>> from cdt.causality.graph import GES
        >>> from cdt.data import load_dataset
        >>> data, graph = load_dataset("sachs")
        >>> obj = GES()
        >>> #The predict() method works without a graph, or with a
        >>> #directed or udirected graph provided as an input
        >>> output = obj.predict(data)    #No graph provided as an argument
        >>>
        >>> output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
        >>>
        >>> output = obj.predict(data, graph)  #With a directed graph
        >>>
        >>> #To view the graph created, run the below commands:
        >>> nx.draw_networkx(output, font_size=8)
        >>> plt.show()
    """

    def __init__(self, score='obs',verbose=None):
        """Init the model and its available arguments."""
        if not RPackages.pcalg:
            raise ImportError("R Package pcalg is not available.")

        super(GES, self).__init__()
        self.scores = {'int': 'GaussL0penIntScore',
                       'obs': 'GaussL0penObsScore'}
        self.arguments = {'{FOLDER}': '/tmp/cdt_ges/',
                          '{FILE}': 'data.csv',
                          '{SKELETON}': 'FALSE',
                          '{GAPS}': 'fixedgaps.csv',
                          '{SCORE}': 'GaussL0penObsScore',
                          '{VERBOSE}': 'FALSE',
                          '{OUTPUT}': 'result.csv'}
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.score = score

    def orient_undirected_graph(self, data, graph):
        """Run GES on an undirected graph.

        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.Graph): Skeleton of the graph to orient

        Returns:
            networkx.DiGraph: Solution given by the GES algorithm.

        """
        # Building setup w/ arguments.
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()
        self.arguments['{SCORE}'] = self.scores[self.score]

        fe = DataFrame(nx.adj_matrix(graph, weight=None).todense())
        fg = DataFrame(1 - fe.values)

        results = self._run_ges(data, fixedGaps=fg, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def orient_directed_graph(self, data, graph):
        """Run GES on a directed graph.

        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.DiGraph): Skeleton of the graph to orient

        Returns:
            networkx.DiGraph: Solution given by the GES algorithm.
        """
        warnings.warn("GES is ran on the skeleton of the given graph.")
        return self.orient_undirected_graph(data, nx.Graph(graph))

    def create_graph_from_data(self, data):
        """Run the GES algorithm.

        Args:
            data (pandas.DataFrame): DataFrame containing the data

        Returns:
            networkx.DiGraph: Solution given by the GES algorithm.

        """
        # Building setup w/ arguments.
        self.arguments['{SCORE}'] = self.scores[self.score]
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()

        results = self._run_ges(data, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def _run_ges(self, data, fixedGaps=None, verbose=True):
        """Setting up and running ges with all arguments."""
        # Run GES
        id = str(uuid.uuid4())
        os.makedirs('/tmp/cdt_ges' + id + '/')
        self.arguments['{FOLDER}'] = '/tmp/cdt_ges' + id + '/'

        def retrieve_result():
            return read_csv('/tmp/cdt_ges' + id + '/result.csv', delimiter=',').values

        try:
            data.to_csv('/tmp/cdt_ges' + id + '/data.csv', header=False, index=False)
            if fixedGaps is not None:
                fixedGaps.to_csv('/tmp/cdt_ges' + id + '/fixedgaps.csv', index=False, header=False)
                self.arguments['{SKELETON}'] = 'TRUE'
            else:
                self.arguments['{SKELETON}'] = 'FALSE'

            ges_result = launch_R_script("{}/R_templates/ges.R".format(os.path.dirname(os.path.realpath(__file__))),
                                         self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_ges' + id + '')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_ges' + id + '/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_ges' + id + '')
        return ges_result
