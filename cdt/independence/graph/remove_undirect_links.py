"""Algorithms to remove indirect links in undirected graphs.

Author: Diviyan Kalainathan and gidonro [github]
"""
import operator
import numpy as np
import networkx as nx
import scipy.stats.mstats as stat
from numpy import linalg as LA


def network_deconvolution(mat, **kwargs):
    """Python implementation/translation of network deconvolution by MIT-KELLIS LAB.

    author:gidonro [Github username](https://github.com/gidonro/Network-Deconvolution)
     LICENSE: MIT-KELLIS LAB
     AUTHORS:
        Algorithm was programmed by Soheil Feizi.
        Paper authors are S. Feizi, D. Marbach,  M. M?©dard and M. Kellis
    Python implementation: Gideon Rosenthal

    REFERENCES:
       For more details, see the following paper:
        Network Deconvolution as a General Method to Distinguish
        Direct Dependencies over Networks
        By: Soheil Feizi, Daniel Marbach,  Muriel Médard and Manolis Kellis
        Nature Biotechnology

    --------------------------------------------------------------------------
     ND.py network deconvolution
    --------------------------------------------------------------------------

    DESCRIPTION:

     INPUT ARGUMENTS:
     mat           Input matrix, if it is a square matrix, the program assumes
                   it is a relevance matrix where mat(i,j) represents the similarity content
                   between nodes i and j. Elements of matrix should be
                   non-negative.
     optional parameters:
     beta          Scaling parameter, the program maps the largest absolute eigenvalue
                   of the direct dependency matrix to beta. It should be
                   between 0 and 1.
     alpha         fraction of edges of the observed dependency matrix to be kept in
                   deconvolution process.
     control       if 0, displaying direct weights for observed
                   interactions, if 1, displaying direct weights for both observed and
                   non-observed interactions.

     OUTPUT ARGUMENTS:

     mat_nd        Output deconvolved matrix (direct dependency matrix). Its components
                   represent direct edge weights of observed interactions.
                   Choosing top direct interactions (a cut-off) depends on the application and
                   is not implemented in this code.

     To apply ND on regulatory networks, follow steps explained in Supplementary notes
     1.4.1 and 2.1 and 2.3 of the paper.
     In this implementation, input matrices are made symmetric.
    """
    alpha = kwargs.get('alpha', 1)
    beta = kwargs.get('beta', 0.99)
    control = kwargs.get('control', 0)

    # ToDO : ASSERTS
    try:
        assert beta < 1 or beta > 0
        assert alpha <= 1 or alpha > 0

    except AssertionError:
        raise ValueError("alpha must be in ]0, 1] and beta in [0, 1]")

    #  Processing the input matrix, diagonal values are filtered
    np.fill_diagonal(mat, 0)

    # Thresholding the input matrix
    y = stat.mquantiles(mat[:], prob=[1 - alpha])
    th = mat >= y
    mat_th = mat * th

    # Making the matrix symetric if already not
    mat_th = (mat_th + mat_th.T) / 2

    # Eigen decomposition
    Dv, U = LA.eigh(mat_th)
    D = np.diag((Dv))
    lam_n = np.abs(np.min(np.min(np.diag(D)), 0))
    lam_p = np.abs(np.max(np.max(np.diag(D)), 0))

    m1 = lam_p * (1 - beta) / beta
    m2 = lam_n * (1 + beta) / beta
    m = max(m1, m2)

    # network deconvolution
    for i in range(D.shape[0]):
        D[i, i] = (D[i, i]) / (m + D[i, i])

    mat_new1 = np.dot(U, np.dot(D, LA.inv(U)))

    # Displying direct weights

    if control == 0:
        ind_edges = (mat_th > 0) * 1.0
        ind_nonedges = (mat_th == 0) * 1.0
        m1 = np.max(np.max(mat * ind_nonedges))
        m2 = np.min(np.min(mat_new1))
        mat_new2 = (mat_new1 + np.max(m1 - m2, 0)) * ind_edges + (mat * ind_nonedges)
    else:
        m2 = np.min(np.min(mat_new1))
        mat_new2 = (mat_new1 + np.max(-m2, 0))

    # linearly mapping the deconvolved matrix to be between 0 and 1
    m1 = np.min(np.min(mat_new2))
    m2 = np.max(np.max(mat_new2))
    mat_nd = (mat_new2 - m1) / (m2 - m1)

    return mat_nd


def clr(M, **kwargs):
    """Implementation of the Context Likelihood or Relatedness Network algorithm.

    Ref:Jeremiah J. Faith, Boris Hayete, Joshua T. Thaden, Ilaria Mogno, Jamey
    Wierzbowski, Guillaume Cottarel, Simon Kasif, James J. Collins, and Timothy
    S. Gardner. Large-scale mapping and validation of escherichia coli
    transcriptional regulation from a compendium of expression profiles.
    PLoS Biology, 2007
    """
    R = np.zeros(M.shape)
    Id = [[0, 0] for i in range(M.shape[0])]
    for i in range(M.shape[0]):
        mu_i = np.mean(M[i, :])
        sigma_i = np.std(M[i, :])
        Id[i] = [mu_i, sigma_i]

    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            z_i = np.max([0, (M[i, j] - Id[i][0]) / Id[i][0]])
            z_j = np.max([0, (M[i, j] - Id[j][0]) / Id[j][0]])
            R[i, j] = np.sqrt(z_i**2 + z_j**2)
            R[j, i] = R[i, j]  # Symmetric

    return R


def aracne(m, **kwargs):
    """Implementation of the ARACNE algorithm.

    Ref: ARACNE: An Algorithm for the Reconstruction of Gene Regulatory Networks in a Mammalian Cellular Context
    Adam A Margolin, Ilya Nemenman, Katia Basso, Chris Wiggins, Gustavo Stolovitzky, Riccardo Dalla Favera and Andrea Califano
    DOI: https://doi.org/10.1186/1471-2105-7-S1-S7
    """
    I0 = kwargs.get('I0', 0.0)  # No default thresholding
    W0 = kwargs.get('W0', 0.05)

    # thresholding
    m = np.where(m > I0, m, 0)

    # Finding triplets and filtering them
    for i in range(m.shape[0]-2):
        for j in range(i+1, m.shape[0]-1):
            for k in range(j+1, m.shape[0]):
                triplet = [m[i, j], m[j, k], m[i, k]]
                min_index, min_value = min(enumerate(triplet), key=operator.itemgetter(1))
                if 0 < min_value < W0:
                    if min_index == 0:
                        m[i, j] = m[j, i] = 0.
                    elif min_index == 1:
                        m[j, k] = m[k, j] = 0.
                    else:
                        m[i, k] = m[k, i] = 0.
    return m


def remove_indirect_links(g, alg="aracne", **kwargs):
    """Apply deconvolution to a graph.

    :param g: nx.Graph to apply deconvolution to
    :param alg: choose which algorithm to use to apply deconvolution
    """
    alg = {"aracne": aracne,
           "nd": network_deconvolution,
           "clr": clr}[alg]
    mat = np.array(nx.adjacency_matrix(g).todense())
    return nx.relabel_nodes(nx.DiGraph(alg(mat, **kwargs)),
                            {idx: i for idx, i in enumerate(list(g.nodes()))})
