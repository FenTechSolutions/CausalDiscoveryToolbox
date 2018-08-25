"""GIES algorithm.
"""
import os
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


class GIES(GraphModel):
    """GIES algorithm.

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

    """

    def __init__(self, score='obs', verbose=False):
        """Init the model and its available arguments."""
        if not RPackages.pcalg:
            raise ImportError("R Package pcalg is not available.")

        super(GIES, self).__init__()
        self.scores = {'int': 'GaussL0penIntScore',
                       'obs': 'GaussL0penObsScore'}
        self.arguments = {'{FILE}': '/tmp/cdt_gies/data.csv',
                          '{SKELETON}': 'FALSE',
                          '{GAPS}': '/tmp/cdt_gies/fixedgaps.csv',
                          '{SCORE}': 'GaussL0penObsScore',
                          '{VERBOSE}': 'FALSE',
                          '{OUTPUT}': '/tmp/cdt_gies/result.csv'}
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.score = score

    def orient_undirected_graph(self, data, graph):
        """Run GIES on an undirected graph.

        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.Graph): Skeleton of the graph to orient

        Returns:
            networkx.DiGraph: Solution given by the GIES algorithm.

        """
        # Building setup w/ arguments.
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()
        self.arguments['{SCORE}'] = self.scores[self.score]

        fe = DataFrame(nx.adj_matrix(graph, weight=None).todense())
        fg = DataFrame(1 - fe.as_matrix())

        results = self._run_gies(data, fixedGaps=fg, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def orient_directed_graph(self, data, graph):
        """Run GIES on a directed_graph.

        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.DiGraph): Skeleton of the graph to orient

        Returns:
            networkx.DiGraph: Solution given by the GIES algorithm.

        """
        warnings.warn("GIES is ran on the skeleton of the given graph.")
        return self.orient_undirected_graph(data, nx.Graph(graph))

    def create_graph_from_data(self, data):
        """Run the GIES algorithm.

        Args:
            data (pandas.DataFrame): DataFrame containing the data

        Returns:
            networkx.DiGraph: Solution given by the GIES algorithm.
        """
        # Building setup w/ arguments.
        self.arguments['{SCORE}'] = self.scores[self.score]
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()

        results = self._run_gies(data, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def _run_gies(self, data, fixedGaps=None, verbose=True):
        """Setting up and running GIES with all arguments."""
        # Run gies
        os.makedirs('/tmp/cdt_gies/')

        def retrieve_result():
            return read_csv('/tmp/cdt_gies/result.csv', delimiter=',').as_matrix()

        try:
            data.to_csv('/tmp/cdt_gies/data.csv', header=False, index=False)
            if fixedGaps is not None:
                fixedGaps.to_csv('/tmp/cdt_gies/fixedgaps.csv', index=False, header=False)
                self.arguments['{SKELETON}'] = 'TRUE'
            else:
                self.arguments['{SKELETON}'] = 'FALSE'

            gies_result = launch_R_script("{}/R_templates/gies.R".format(os.path.dirname(os.path.realpath(__file__))),
                                          self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_gies')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_gies/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_gies')
        return gies_result
