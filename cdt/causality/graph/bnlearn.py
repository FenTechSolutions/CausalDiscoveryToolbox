"""BN learn algorithms.

Imported from the bnlearn package.
Author: Diviyan Kalainathan
"""
import os
import warnings
import networkx as nx
from shutil import rmtree
from .model import GraphModel
from pandas import DataFrame, read_csv
from ...utils.R import RPackages, launch_R_script


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class BNlearnAlgorithm(GraphModel):
    """BNlearn algorithm.

    Available tests:
    • discrete case (categorical variables)
        – mutual information: an information-theoretic distance measure.
          It's proportional to the log-likelihood ratio (they differ by a 2n factor)
          and is related to the deviance of the tested models. The asymptotic χ2 test
          (mi and mi-adf,  with  adjusted  degrees  of  freedom), the Monte Carlo
          permutation test (mc-mi), the sequential Monte Carlo permutation
          test (smc-mi), and the semiparametric test (sp-mi) are implemented.
        – shrinkage estimator for the mutual information (mi-sh): an improved
          asymptotic χ2 test based on the James-Stein estimator for the mutual
          information.
        – Pearson’s X2 : the classical Pearson's X2 test for contingency tables.
          The asymptotic χ2 test (x2 and x2-adf, with adjusted degrees of freedom),
          the Monte Carlo permutation test (mc-x2), the sequential Monte Carlo
          permutation test (smc-x2) and semiparametric test (sp-x2) are implemented.

    • discrete case (ordered factors)
        – Jonckheere-Terpstra : a trend test for ordinal variables.  The
          asymptotic normal test (jt), the Monte Carlo permutation test (mc-jt)
          and the sequential Monte Carlo permutation test (smc-jt) are implemented.

    • continuous case (normal variables)
        – linear  correlation:  Pearson’s  linear  correlation.   The  exact
          Student’s  t  test  (cor),  the Monte Carlo permutation test (mc-cor)
          and the sequential Monte Carlo permutation test (smc-cor) are implemented.
        – Fisher’s Z: a transformation of the linear correlation with asymptotic
          normal distribution. Used by commercial software (such as TETRAD II)
          for the PC algorithm (an R implementation is present in the pcalg
          package on CRAN). The asymptotic normal test (zf), the Monte Carlo
          permutation test (mc-zf) and the sequential Monte Carlo permutation
          test (smc-zf) are implemented.
        – mutual information: an information-theoretic distance measure.  Again
          it is proportional to the log-likelihood ratio (they differ by a 2n
          factor). The asymptotic χ2 test (mi-g), the Monte Carlo permutation
          test (mc-mi-g) and the sequential Monte Carlo permutation test
          (smc-mi-g) are implemented.

        – shrinkage estimator for the mutual information(mi-g-sh): an improved
          asymptotic χ2 test based on the James-Stein estimator for the mutual
          information.

    • hybrid case (mixed discrete and normal variables)
        – mutual information: an information-theoretic distance measure.  Again
          it is proportionalto the log-likelihood ratio (they differ by a 2n
          factor). Only the asymptotic χ2 test (mi-cg) is implemented."""

    def __init__(self):
        """Init the model."""
        if not RPackages.bnlearn:
            raise ImportError("R Package bnlearn is not available.")
        super(BNlearnAlgorithm, self).__init__()
        self.arguments = {'{FILE}': '/tmp/cdt_bnlearn/data.csv',
                          '{SKELETON}': 'FALSE',
                          '{ALGORITHM}': None,
                          '{WHITELIST}': '/tmp/cdt_bnlearn/whitelist.csv',
                          '{BLACKLIST}': '/tmp/cdt_bnlearn/blacklist.csv',
                          '{SCORE}': 'NULL',
                          '{OPTIM}': 'FALSE',
                          '{ALPHA}': '0.05',
                          '{BETA}': 'NULL',
                          '{VERBOSE}': 'FALSE',
                          '{OUTPUT}': '/tmp/cdt_bnlearn/result.csv'}

    def orient_undirected_graph(self, data, graph, score='NULL', alpha=0.05, beta='NULL',
                                optim=False, verbose=False, **kwargs):
        """Run the algorithm on an undirected graph."""
        # Building setup w/ arguments.
        self.arguments['{VERBOSE}'] = str(verbose).upper()
        self.arguments['{SCORE}'] = score
        self.arguments['{BETA}'] = str(beta)
        self.arguments['{OPTIM}'] = str(optim).upper()
        self.arguments['{ALPHA}'] = str(alpha)

        whitelist = DataFrame(list(nx.edges(graph)), columns=["from", "to"])
        blacklist = DataFrame(list(nx.edges(nx.DiGraph(DataFrame(-nx.adj_matrix(graph, weight=None).to_dense() + 1,
                                                                 columns=list(graph.nodes()),
                                                                 index=list(graph.nodes()))))), columns=["from", "to"])
        results = self.run_bnlearn(data, whitelist=whitelist,
                                   blacklist=blacklist, verbose=verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def orient_directed_graph(self, data, graph, *args, **kwargs):
        """Run the algorithm on a directed_graph."""
        warnings.warn("The algorithm is ran on the skeleton of the given graph.")
        return self.orient_undirected_graph(data, nx.Graph(graph), *args, **kwargs)

    def create_graph_from_data(self, data, score='NULL', alpha=0.05, beta='NULL',
                               optim=False, verbose=False, **kwargs):
        """Run the algorithm on data.

        :param data: DataFrame containing the data
        :param score: score used for bnlearn. ['obs', 'int']
        :param verbose: if TRUE, detailed output is provided.
        """
        # Building setup w/ arguments.
        self.arguments['{SCORE}'] = score
        self.arguments['{VERBOSE}'] = str(verbose).upper()
        self.arguments['{BETA}'] = str(beta)
        self.arguments['{OPTIM}'] = str(optim).upper()
        self.arguments['{ALPHA}'] = str(alpha)

        results = self.run_bnlearn(data, verbose=verbose)
        graph = nx.DiGraph()
        graph.add_edges_from(results)
        return graph

    def run_bnlearn(self, data, whitelist=None, blacklist=None, verbose=True):
        """Setting up and running bnlearn with all arguments."""
        # Run the algorithm
        os.makedirs('/tmp/cdt_bnlearn/')

        def retrieve_result():
            return read_csv('/tmp/cdt_bnlearn/result.csv', delimiter=',').as_matrix()

        try:
            data.to_csv('/tmp/cdt_bnlearn/data.csv', index=False)
            if blacklist is not None:
                whitelist.to_csv('/tmp/cdt_bnlearn/whitelist.csv', index=False, header=False)
                blacklist.to_csv('/tmp/cdt_bnlearn/blacklist.csv', index=False, header=False)
                self.arguments['{SKELETON}'] = 'TRUE'
            else:
                self.arguments['{SKELETON}'] = 'FALSE'

            bnlearn_result = launch_R_script("{}/R_templates/bnlearn.R".format(os.path.dirname(os.path.realpath(__file__))),
                                             self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_bnlearn')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_bnlearn/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_bnlearn')
        return bnlearn_result


class GS(BNlearnAlgorithm):
    """Grow-Shrink algorithm.

    Margaritis D (2003).
    Learning Bayesian Network Model Structure from Data
    . Ph.D. thesis, School
    of Computer Science, Carnegie-Mellon University, Pittsburgh, PA. Available as Technical Report
    CMU-CS-03-153
    """

    def __init__(self):
        """Init the model."""
        super(GS, self).__init__()
        self.arguments['{ALGORITHM}'] = 'gs'


class IAMB(BNlearnAlgorithm):
    """IAMB algorithm.

    Tsamardinos  I,  Aliferis  CF,  Statnikov  A  (2003).   "Algorithms  for  Large  Scale  Markov  Blanket
    Discovery".  In "Proceedings of the Sixteenth International Florida Artificial Intelligence Research
    Society Conference", pp. 376-381. AAAI Press.
    """

    def __init__(self):
        """Init the model."""
        super(IAMB, self).__init__()
        self.arguments['{ALGORITHM}'] = 'iamb'


class Fast_IAMB(BNlearnAlgorithm):
    """Fast IAMB algorithm.

    Yaramakala S, Margaritis D (2005).  "Speculative Markov Blanket Discovery for Optimal Feature
    Selection".  In "ICDM ’05:  Proceedings of the Fifth IEEE International Conference on Data Min-
    ing", pp. 809-812. IEEE Computer Society.
    """

    def __init__(self):
        """Init the model."""
        super(Fast_IAMB, self).__init__()
        self.arguments['{ALGORITHM}'] = 'fast.iamb'


class Inter_IAMB(BNlearnAlgorithm):
    """Inter IAMB algorithm.

    Yaramakala S, Margaritis D (2005).  "Speculative Markov Blanket Discovery for Optimal Feature
    Selection".  In "ICDM ’05:  Proceedings of the Fifth IEEE International Conference on Data Min-
    ing", pp. 809-812. IEEE Computer Society.
    """

    def __init__(self):
        """Init the model."""
        super(Inter_IAMB, self).__init__()
        self.arguments['{ALGORITHM}'] = 'inter.iamb'


class MMPC(BNlearnAlgorithm):
    """Max-Min Parents-Children algorithm.

    Tsamardinos I, Aliferis CF, Statnikov A (2003). "Time and Sample Efficient Discovery of Markov
    Blankets and Direct Causal Relations".  In "KDD ’03:  Proceedings of the Ninth ACM SIGKDD
    International Conference on Knowledge Discovery and Data Mining", pp. 673-678. ACM.
    Tsamardinos I, Brown LE, Aliferis CF (2006).  "The Max-Min Hill-Climbing Bayesian Network
    Structure Learning Algorithm".
    Machine Learning,65(1), 31-78.
    """

    def __init__(self):
        """Init the model."""
        super(MMPC, self).__init__()
        self.arguments['{ALGORITHM}'] = 'mmpc'
