"""CCDR algorithm.

Imported from the Pcalg package.
Author: Diviyan Kalainathan
"""
import os
import uuid
import warnings
import networkx as nx
from shutil import rmtree
from .model import GraphModel
from pandas import read_csv
from ...utils.Settings import SETTINGS
from ...utils.R import RPackages, launch_R_script


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class CCDr(GraphModel):
    r"""CCDr algorithm.
    Concave penalized Coordinate Descent with reparametrization) structure
    learning algorithm as described in Aragam and Zhou (2015). This is a fast,
    score based method for learning Bayesian networks that uses sparse
    regularization and block-cyclic coordinate descent.

    Imported from the 'sparsebn' package.

    .. warning::
       This implementation of CCDr does not support starting with a graph.
    """

    def __init__(self, verbose=None):
        """Init the model and its available arguments."""
        if not RPackages.sparsebn:
            raise ImportError("R Package sparsebn is not available.")

        super(CCDr, self).__init__()
        self.arguments = {'{FOLDER}': '/tmp/cdt_CCDR/',
                          '{FILE}': 'data.csv',
                          '{VERBOSE}': 'FALSE',
                          '{OUTPUT}': 'result.csv'}
        # ToDo self.alpha = 0
        self.verbose = SETTINGS.get_default(verbose=verbose)

    def orient_undirected_graph(self, data, graph,
                                verbose=False, **kwargs):
        """Run CCDr on an undirected graph."""
        # Building setup w/ arguments.
        raise ValueError("CCDR cannot (yet) be ran with a skeleton/directed graph.")

    def orient_directed_graph(self, data, graph, *args, **kwargs):
        """Run CCDR on a directed_graph."""
        raise ValueError("CCDR cannot (yet) be ran with a skeleton/directed graph.")

    def create_graph_from_data(self, data, **kwargs):
        """Apply causal discovery on observational data using CCDr.

        Args:
            data (pandas.DataFrame): DataFrame containing the data

        Returns:
            networkx.DiGraph: Solution given by the CCDR algorithm.
        """
        # Building setup w/ arguments.
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()
        results = self._run_ccdr(data, verbose=self.verbose)
        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def _run_ccdr(self, data, fixedGaps=None, verbose=True):
        """Setting up and running CCDr with all arguments."""
        # Run CCDr
        id = str(uuid.uuid4())
        os.makedirs('/tmp/cdt_CCDR' + id + '/')
        self.arguments['{FOLDER}'] = '/tmp/cdt_CCDR' + id + '/'

        def retrieve_result():
            return read_csv('/tmp/cdt_CCDR' + id + '/result.csv', delimiter=',').values

        try:
            data.to_csv('/tmp/cdt_CCDR' + id + '/data.csv', header=False, index=False)
            ccdr_result = launch_R_script("{}/R_templates/CCDr.R".format(os.path.dirname(os.path.realpath(__file__))),
                                         self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_CCDR' + id + '')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_CCDR' + id + '/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_CCDR' + id + '')
        return ccdr_result
