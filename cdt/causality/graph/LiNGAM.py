"""LiNGAM algorithm.

Imported from the Pcalg package.
Author: Diviyan Kalainathan
"""
import os
import warnings
import networkx as nx
from shutil import rmtree
from .model import GraphModel
from pandas import read_csv
from ...utils.R import RPackages, launch_R_script
from ...utils.Settings import SETTINGS


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class LiNGAM(GraphModel):
    r"""LiNGAM algorithm.
    
    Args:
        verbose (bool): Sets the verbosity of the algorithm. Defaults to 
           `cdt.SETTINGS.verbose`

    .. note::
       Ref: S.  Shimizu,  P.O.  Hoyer,  A.  Hyvärinen,  A.  Kerminen  (2006)
       A  Linear  Non-Gaussian  Acyclic Model for Causal Discovery;
       Journal of Machine Learning Research 7, 2003–2030.

    .. warning::
       This implementation of LiNGAM does not support starting with a graph.

    """

    def __init__(self, verbose=False):
        """Init the model and its available arguments."""
        if not RPackages.pcalg:
            raise ImportError("R Package pcalg is not available.")

        super(LiNGAM, self).__init__()

        self.arguments = {'{FILE}': '/tmp/cdt_LiNGAM/data.csv',
                          '{VERBOSE}': 'FALSE',
                          '{OUTPUT}': '/tmp/cdt_LiNGAM/result.csv'}
        self.verbose = SETTINGS.get_default(verbose=verbose)

    def orient_undirected_graph(self, data, graph):
        """Run LiNGAM on an undirected graph."""
        # Building setup w/ arguments.
        raise ValueError("LiNGAM cannot (yet) be ran with a skeleton/directed graph.")

    def orient_directed_graph(self, data, graph):
        """Run LiNGAM on a directed_graph."""
        raise ValueError("LiNGAM cannot (yet) be ran with a skeleton/directed graph.")

    def create_graph_from_data(self, data):
        """Run the LiNGAM algorithm.

        Args:
            data (pandas.DataFrame): DataFrame containing the data

        Returns:
            networkx.DiGraph: Solution given by the LiNGAM algorithm.

        """
        # Building setup w/ arguments.
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()
        results = self._run_LiNGAM(data, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def _run_LiNGAM(self, data, fixedGaps=None, verbose=True):
        """Setting up and running LiNGAM with all arguments."""
        # Run LiNGAM
        os.makedirs('/tmp/cdt_LiNGAM/')

        def retrieve_result():
            return read_csv('/tmp/cdt_LiNGAM/result.csv', delimiter=',').as_matrix()

        try:
            data.to_csv('/tmp/cdt_LiNGAM/data.csv', header=False, index=False)
            lingam_result = launch_R_script("{}/R_templates/lingam.R".format(os.path.dirname(os.path.realpath(__file__))),
                                            self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_LiNGAM')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_LiNGAM/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_LiNGAM')
        return lingam_result
