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

    Args:
        score (str): Score used to fit the gaussian processes.
        cutoff (float): threshold value for variable selection.
        variablesel (bool): Perform a variable selection step.
        selmethod (str): Method used for variable selection.
        pruning (bool): Perform an initial pruning step.
        prunmethod (str): Method used for pruning.
        nb_jobs (int): Number of jobs to run in parallel.
        verbose (bool): Sets the verbosity of the output.

    Available scores:
       + nonlinear: 'SEMGAM'
       + linear: 'SEMLIN'

    Available variable selection methods:
       + gamboost': 'selGamBoost'
       + gam': 'selGam'
       + lasso': 'selLasso'
       + linear': 'selLm'
       + linearboost': 'selLmBoost'

    Default Parameters:
       + FILE: '/tmp/cdt_CAM/data.csv'
       + SCORE: 'SEMGAM'
       + VARSEL: 'TRUE'
       + SELMETHOD: 'selGamBoost'
       + PRUNING: 'TRUE'
       + PRUNMETHOD: 'selGam'
       + NJOBS: str(SETTINGS.NB_JOBS)
       + CUTOFF: str(0.001)
       + VERBOSE: 'FALSE'
       + OUTPUT: '/tmp/cdt_CAM/result.csv'

    .. note::
       Ref:
       J. Peters, J. Mooij, D. Janzing, B. Schölkopf:
       Causal Discovery with Continuous Additive Noise Models,
       JMLR 15:2009-2053, 2014.

    .. warning::
       This implementation of CAM does not support starting with a graph.
       The adaptation will be made at a later date.
    """

    def __init__(self, score='nonlinear', cutoff=0.001, variablesel=True,
                 selmethod='gamboost', pruning=False, prunmethod='gam',
                 nb_jobs=None, verbose=None):
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
        self.score = score
        self.cutoff = cutoff
        self.variablesel = variablesel
        self.selmethod = selmethod
        self.pruning = pruning
        self.prunmethod = prunmethod
        self.nb_jobs = SETTINGS.get_default(nb_jobs=nb_jobs)
        self.verbose = SETTINGS.get_default(verbose=verbose)

    def orient_undirected_graph(self, data, graph, score='obs',
                                verbose=False, **kwargs):
        """Run CAM on an undirected graph."""
        # Building setup w/ arguments.
        raise ValueError("CAM cannot (yet) be ran with a skeleton/directed graph.")

    def orient_directed_graph(self, data, graph, *args, **kwargs):
        """Run CAM on a directed_graph."""
        raise ValueError("CAM cannot (yet) be ran with a skeleton/directed graph.")

    def create_graph_from_data(self, data, **kwargs):
        """Apply causal discovery on observational data using CAM.

        Args:
            data (pandas.DataFrame): DataFrame containing the data

        Returns:
            networkx.DiGraph: Solution given by the CAM algorithm.
        """
        # Building setup w/ arguments.
        self.arguments['{SCORE}'] = self.scores[self.score]
        self.arguments['{CUTOFF}'] = str(self.cutoff)
        self.arguments['{VARSEL}'] = str(self.variablesel).upper()
        self.arguments['{SELMETHOD}'] = self.var_selection[self.selmethod]
        self.arguments['{PRUNING}'] = str(self.pruning).upper()
        self.arguments['{PRUNMETHOD}'] = self.var_selection[self.prunmethod]
        self.arguments['{NJOBS}'] = str(self.nb_jobs)
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()
        results = self._run_cam(data, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def _run_cam(self, data, fixedGaps=None, verbose=True):
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
