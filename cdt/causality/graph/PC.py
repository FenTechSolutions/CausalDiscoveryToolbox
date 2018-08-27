"""PC algorithm by C.Glymour & P.Sprites (REF, 2000).

Imported from the Pcalg package.
Author = Diviyan Kalainathan
"""
import os
import warnings
import networkx as nx
from shutil import rmtree
from .model import GraphModel
from pandas import DataFrame, read_csv
from ...utils.Settings import SETTINGS
from ...utils.R import RPackages, launch_R_script


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class PC(GraphModel):
    """PC algorithm by C.Glymour & P.Sprites.

    Args:
        CItest (str): Test for conditional independence.
        method (str): Heuristic for testing CI.
        alpha (float): significance level (number in (0, 1) for the individual
           conditional independence tests.
        njobs (int): number of processor cores to use for parallel computation. 
           Only available for method = "stable.fast" (set as default).
        verbose: if TRUE, detailed output is provided.

    Attributes:
        arguments (dict): contains all current parameters used in the PC
           algorithm execution.
        dir_CI_test (dict): contains all available conditional independence
           tests.
        dir_method_indep (dict): contains all available heuristics for CI
           testing.

    Available conditional independence tests:
        + gaussian: "pcalg::gaussCItest"
        + hsic: "kpcalg::kernelCItest"
        + discrete: "pcalg::disCItest"
        + binary: "pcalg::binCItest"

    Available heuristics for CI testing:
        + dcc: "data=X, ic.method=\"dcc\""
        + hsic_gamma: "data=X, ic.method=\"hsic.gamma\""
        + hsic_perm: "data=X, ic.method=\"hsic.perm\""
        + hsic_clus: "data=X, ic.method=\"hsic.clus\""
        + corr: "C = cor(X), n = nrow(X)"

    Default Parameters:
        + FILE: '/tmp/cdt_pc/data.csv'
        + SKELETON: 'FALSE'
        + EDGES: '/tmp/cdt_pc/fixededges.csv'
        + GAPS: '/tmp/cdt_pc/fixedgaps.csv'
        + CITEST: "pcalg::gaussCItest"
        + METHOD_INDEP: "C = cor(X), n = nrow(X)"
        + SELMAT: 'NULL'
        + DIRECTED: 'TRUE'
        + SETOPTIONS: 'NULL'
        + ALPHA: '0.01'
        + VERBOSE: 'FALSE'
        + OUTPUT: '/tmp/cdt_pc/result.csv'

    .. note::
       Ref:
       D.Colombo and M.H. Maathuis (2014).
       Order-independent constraint-based causal structure learning.
       Journal of Machine Learning Research 15 3741-3782.

       M. Kalisch, M. Maechler, D. Colombo, M.H. Maathuis and P. Buehlmann (2012).
       Causal Inference Using Graphical Models with the R Package pcalg.
       Journal of Statistical Software 47(11) 1–26, http://www.jstatsoft.org/v47/i11/

       M. Kalisch and P. Buehlmann (2007).
       Estimating high-dimensional directed acyclic graphs with the PC-algorithm.
       JMLR 8 613-636.

       J. Ramsey, J. Zhang and P. Spirtes (2006).
       Adjacency-faithfulness and conservative causal inference.
       In Proceedings of the 22nd Annual Conference on Uncertainty in Artificial
       Intelligence. AUAI Press, Arlington, VA.

       P. Spirtes, C. Glymour and R. Scheines (2000).
       Causation, Prediction, and Search, 2nd edition. The MIT Press

       Imported from the Pcalg package.
    """

    def __init__(self, CItest="gaussian", method_indep='corr', alpha=0.01,
                 nb_jobs=None, verbose=None):
        """Init the model and its available arguments."""
        if not (RPackages.pcalg and RPackages.kpcalg):
            raise ImportError("R Package (k)pcalg is not available.")

        super(PC, self).__init__()
        self.dir_CI_test = {"gaussian": "pcalg::gaussCItest",
                            "hsic": "kpcalg::kernelCItest",
                            "discrete": "pcalg::disCItest",
                            "binary": "pcalg::binCItest"}
        self.dir_method_indep = {'dcc': "data=X, ic.method=\"dcc\"",
                                 'hsic_gamma': "data=X, ic.method=\"hsic.gamma\"",
                                 'hsic_perm': "data=X, ic.method=\"hsic.perm\"",
                                 'hsic_clus': "data=X, ic.method=\"hsic.clus\"",
                                 'corr': "C = cor(X), n = nrow(X)"}
        self.CI_test = CItest 
        self.method_indep = method_indep
        self.alpha = alpha
        self.nb_jobs = SETTINGS.get_default(nb_jobs=nb_jobs)
        self.verbose = SETTINGS.get_default(verbose=verbose)
        # Define default args
        self.arguments = {'{FILE}': '/tmp/cdt_pc/data.csv',
                          '{SKELETON}': 'FALSE',
                          '{EDGES}': '/tmp/cdt_pc/fixededges.csv',
                          '{GAPS}': '/tmp/cdt_pc/fixedgaps.csv',
                          '{CITEST}': "pcalg::gaussCItest",
                          '{METHOD_INDEP}': "C = cor(X), n = nrow(X)",
                          '{SELMAT}': 'NULL',
                          '{DIRECTED}': 'TRUE',
                          '{SETOPTIONS}': 'NULL',
                          '{ALPHA}': '',
                          '{VERBOSE}': 'FALSE',
                          '{OUTPUT}': '/tmp/cdt_pc/result.csv'}

    def orient_undirected_graph(self, data, graph, **kwargs):
        """Run PC on an undirected graph.

        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.Graph): Skeleton of the graph to orient

        Returns:
            networkx.DiGraph: Solution given by PC on the given skeleton.
        """
        # Building setup w/ arguments.
        self.arguments['{CITEST}'] = self.dir_CI_test[self.CI_test]
        self.arguments['{METHOD_INDEP}'] = self.dir_method_indep[self.method_indep]
        self.arguments['{DIRECTED}'] = 'TRUE'
        self.arguments['{ALPHA}'] = str(self.alpha)
        self.arguments['{NJOBS}'] = str(self.nb_jobs)
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()

        fe = DataFrame(nx.adj_matrix(graph, weight=None).todense())
        fg = DataFrame(1 - fe.as_matrix())

        results = self._run_pc(data, fixedEdges=fe, fixedGaps=fg, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def orient_directed_graph(self, data, graph, *args, **kwargs):
        """Run PC on a directed_graph (Only takes account of the skeleton of
        the graph).

        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.DiGraph): Skeleton of the graph to orient

        Returns:
            networkx.DiGraph: Solution given by PC on the given skeleton.

        .. warning::
           The algorithm is ran on the skeleton of the given graph.

        """
        warnings.warn("PC is ran on the skeleton of the given graph.")
        return self.orient_undirected_graph(data, nx.Graph(graph), *args, **kwargs)

    def create_graph_from_data(self, data, **kwargs):
        """Run the PC algorithm.

        Args:
            data (pandas.DataFrame): DataFrame containing the data

        Returns:
            networkx.DiGraph: Solution given by PC on the given data.
       """
        # Building setup w/ arguments.
        self.arguments['{CITEST}'] = self.dir_CI_test[self.CI_test]
        self.arguments['{METHOD_INDEP}'] = self.dir_method_indep[self.method_indep]
        self.arguments['{DIRECTED}'] = 'TRUE'
        self.arguments['{ALPHA}'] = str(self.alpha)
        self.arguments['{NJOBS}'] = str(self.nb_jobs)
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()

        results = self._run_pc(data, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def _run_pc(self, data, fixedEdges=None, fixedGaps=None, verbose=True):
        """Setting up and running pc with all arguments."""
        # Checking coherence of arguments
        if (self.arguments['{CITEST}'] == self.dir_CI_test['hsic']
           and self.arguments['{METHOD_INDEP}'] == self.dir_method_indep['corr']):
            warnings.warn('Selected method for indep is unfit for the hsic test,'
                          ' setting the hsic.gamma method.')
            self.arguments['{METHOD_INDEP}'] = self.dir_method_indep['hsic_gamma']

        elif (self.arguments['{CITEST}'] != self.dir_CI_test['hsic']
              and self.arguments['{METHOD_INDEP}'] != self.dir_method_indep['corr']):
            warnings.warn('Selected method for indep is unfit for the selected test,'
                          ' setting the classic correlation-based method.')
            self.arguments['{METHOD_INDEP}'] = self.dir_method_indep['corr']

        # Run PC
        os.makedirs('/tmp/cdt_pc/')

        def retrieve_result():
            return read_csv('/tmp/cdt_pc/result.csv', delimiter=',').as_matrix()

        try:
            data.to_csv('/tmp/cdt_pc/data.csv', header=False, index=False)
            if fixedGaps is not None and fixedEdges is not None:
                fixedGaps.to_csv('/tmp/cdt_pc/fixedgaps.csv', index=False, header=False)
                fixedEdges.to_csv('/tmp/cdt_pc/fixededges.csv', index=False, header=False)
                self.arguments['{SKELETON}'] = 'TRUE'
            else:
                self.arguments['{SKELETON}'] = 'FALSE'

            pc_result = launch_R_script("{}/R_templates/pc.R".format(os.path.dirname(os.path.realpath(__file__))),
                                        self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_pc')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_pc/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_pc')
        return pc_result
