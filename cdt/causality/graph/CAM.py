"""CAM algorithm.

Imported from the Pcalg package.
Author: Diviyan Kalainathan
"""
import os
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


class CAM(GraphModel):
    r"""CAM algorithm.

    Ref:
    J. Peters, J. Mooij, D. Janzing, B. Sch\"olkopf:
    Causal Discovery with Continuous Additive Noise Models,
    JMLR 15:2009-2053, 2014.
    """

    def __init__(self):
        """Init the model and its available arguments."""
        if not RPackages.CAM:
            raise ImportError("R Package CAM is not available.")

        super(CAM, self).__init__()
        self.scores = {'nonlinear': 'SEMGAM',
                       'linear': 'SEMLIN'}
        self.var_selection = {'gamboost': 'selGamBoost',
                              'gam': 'selGam',
                              'lasso': 'selLasso',
                              'linear': 'selLm',
                              'linearboost': 'selLmBoost'}
        self.arguments = {'{FILE}': '/tmp/cdt_CAM/data.csv',
                          '{SCORE}': 'SEMGAM',
                          '{VARSEL}': 'TRUE',
                          '{SELMETHOD}': 'selGamBoost',
                          '{PRUNING}': 'TRUE',
                          '{PRUNMETHOD}': 'selGam',
                          '{NJOBS}': str(SETTINGS.NB_JOBS),
                          '{CUTOFF}': str(0.001),
                          '{VERBOSE}': 'FALSE',
                          '{OUTPUT}': '/tmp/cdt_CAM/result.csv'}

    def orient_undirected_graph(self, data, graph, score='obs',
                                verbose=False, **kwargs):
        """Run CAM on an undirected graph."""
        # Building setup w/ arguments.
        raise ValueError("CAM cannot (yet) be ran with a skeleton/directed graph.")

    def orient_directed_graph(self, data, graph, *args, **kwargs):
        """Run CAM on a directed_graph."""
        raise ValueError("CAM cannot (yet) be ran with a skeleton/directed graph.")

    def create_graph_from_data(self, data, score='nonlinear', cutoff=0.001, variablesel=True,
                               selmethod='gamboost', pruning=False, prunmethod='gam',
                               njobs=SETTINGS.NB_JOBS, verbose=False, **kwargs):
        """Run the CAM algorithm.

        :param data: DataFrame containing the data
        :param score: score used for CAM.
        :param verbose: if TRUE, detailed output is provided.
        """
        # Building setup w/ arguments.
        self.arguments['{SCORE}'] = self.scores[score]
        self.arguments['{CUTOFF}'] = str(cutoff)
        self.arguments['{VARSEL}'] = str(variablesel).upper()
        self.arguments['{SELMETHOD}'] = self.var_selection[selmethod]
        self.arguments['{PRUNING}'] = str(pruning).upper()
        self.arguments['{PRUNMETHOD}'] = self.var_selection[prunmethod]
        self.arguments['{NJOBS}'] = str(njobs)
        self.arguments['{VERBOSE}'] = str(verbose).upper()
        results = self.run_CAM(data, verbose=verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def run_CAM(self, data, fixedGaps=None, verbose=True):
        """Setting up and running CAM with all arguments."""
        # Run CAM
        os.makedirs('/tmp/cdt_CAM/')

        def retrieve_result():
            return read_csv('/tmp/cdt_CAM/result.csv', delimiter=',').as_matrix()

        try:
            data.to_csv('/tmp/cdt_CAM/data.csv', header=False, index=False)
            cam_result = launch_R_script("{}/R_templates/cam.R".format(os.path.dirname(os.path.realpath(__file__))),
                                         self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_CAM')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_CAM/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_CAM')
        return cam_result
