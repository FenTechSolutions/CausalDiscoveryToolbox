"""Utilities for graph not included in Networkx.

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

import networkx as nx
from copy import deepcopy
import operator
import numpy as np
import scipy.stats.mstats as stat
from numpy import linalg as LA


def network_deconvolution(mat, **kwargs):
    """Python implementation/translation of network deconvolution by MIT-KELLIS LAB.

    .. note::
       For networkx graphs, use the cdt.utils.graph.remove_indirect_links function
       code author:gidonro [Github username](https://github.com/gidonro/Network-Deconvolution)

       LICENSE: MIT-KELLIS LAB

       AUTHORS:
       Algorithm was programmed by Soheil Feizi.
       Paper authors are S. Feizi, D. Marbach,  M. M?©dard and M. Kellis
       Python implementation: Gideon Rosenthal

       For more details, see the following paper:
       Network Deconvolution as a General Method to Distinguish
       Direct Dependencies over Networks

       By: Soheil Feizi, Daniel Marbach,  Muriel Médard and Manolis Kellis
       Nature Biotechnology

    Args:
         mat (numpy.ndarray): matrix, if it is a square matrix, the program assumes
             it is a relevance matrix where mat(i,j) represents the similarity content
             between nodes i and j. Elements of matrix should be
             non-negative.
         beta (float): Scaling parameter, the program maps the largest absolute eigenvalue
             of the direct dependency matrix to beta. It should be
             between 0 and 1.
         alpha (float): fraction of edges of the observed dependency matrix to be kept in
             deconvolution process.
         control (int): if 0, displaying direct weights for observed
             interactions, if 1, displaying direct weights for both observed and
             non-observed interactions.

    Returns:
        numpy.ndarray: Output deconvolved matrix (direct dependency matrix). Its components
        represent direct edge weights of observed interactions.
        Choosing top direct interactions (a cut-off) depends on the application and
        is not implemented in this code.

    Example:
        >>> from cdt.utils.graph import network_deconvolution
        >>> import networkx as nx
        >>> # Generate sample data
        >>> from cdt.data import AcyclicGraphGenerator
        >>> graph = AcyclicGraphGenerator(linear).generate()[1]
        >>> adj_mat = nx.adjacency_matrix(graph).todense()
        >>> output = network_deconvolution(adj_mat)

     .. note::
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

    .. note::
       For networkx graphs, use the cdt.utils.graph.remove_indirect_links function

    Args:
        mat (numpy.ndarray): matrix, if it is a square matrix, the program assumes
            it is a relevance matrix where mat(i,j) represents the similarity content
            between nodes i and j. Elements of matrix should be
            non-negative.

    Returns:
        numpy.ndarray: Output deconvolved matrix (direct dependency matrix). Its components
        represent direct edge weights of observed interactions.

    Example:
        >>> from cdt.utils.graph import clr
        >>> import networkx as nx
        >>> # Generate sample data
        >>> from cdt.data import AcyclicGraphGenerator
        >>> graph = AcyclicGraphGenerator(linear).generate()[1]
        >>> adj_mat = nx.adjacency_matrix(graph).todense()
        >>> output = clr(adj_mat)

    .. note::
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

    .. note::
       For networkx graphs, use the cdt.utils.graph.remove_indirect_links function

    Args:
        mat (numpy.ndarray): matrix, if it is a square matrix, the program assumes
            it is a relevance matrix where mat(i,j) represents the similarity content
            between nodes i and j. Elements of matrix should be
            non-negative.

    Returns:
        numpy.ndarray: Output deconvolved matrix (direct dependency matrix). Its components
        represent direct edge weights of observed interactions.

    Example:
        >>> from cdt.utils.graph import aracne
        >>> import networkx as nx
        >>> # Generate sample data
        >>> from cdt.data import AcyclicGraphGenerator
        >>> graph = AcyclicGraphGenerator(linear).generate()[1]
        >>> adj_mat = nx.adjacency_matrix(graph).todense()
        >>> output = aracne(adj_mat)

    .. note::
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
    """Apply deconvolution to a networkx graph.

    Args:
       g (networkx.Graph): Graph to apply deconvolution to
       alg (str): Algorithm to use ('aracne', 'clr', 'nd')
       kwargs (dict): extra options for algorithms

    Returns:
       networkx.Graph: graph with undirected links removed.

    Example:
        >>> from cdt.utils.graph import remove_indirect_links
        >>> import networkx as nx
        >>> # Generate sample data
        >>> from cdt.data import AcyclicGraphGenerator
        >>> graph = AcyclicGraphGenerator(linear).generate()[1]
        >>> output = remove_indirect_links(graph, alg='aracne')
    """
    alg = {"aracne": aracne,
           "nd": network_deconvolution,
           "clr": clr}[alg]
    order_list = list(g.nodes())
    mat = np.array(nx.adjacency_matrix(g, nodelist=order_list).todense())
    return nx.relabel_nodes(nx.DiGraph(alg(mat, **kwargs)),
                            {idx: i for idx, i in enumerate(order_list)})


def dagify_min_edge(g):
    """Input a graph and output a DAG.

    The heuristic is to reverse the edge with the lowest score of the cycle
    if possible, else remove it.

    Args:
        g (networkx.DiGraph): Graph to modify to output a DAG

    Returns:
        networkx.DiGraph: DAG made out of the input graph.

    Example:
        >>> from cdt.utils.graph import dagify_min_edge
        >>> import networkx as nx
        >>> import numpy as np
        >>> # Generate sample data
        >>> graph = nx.DiGraph((np.ones(4) - np.eye(4)) *
                               np.random.uniform(size=(4,4)))
        >>> output = dagify_min_edge(graph)
    """
    ncycles = len(list(nx.simple_cycles(g)))
    while not nx.is_directed_acyclic_graph(g):
        cycle = next(nx.simple_cycles(g))
        edges = [(cycle[-1], cycle[0])]
        scores = [(g[cycle[-1]][cycle[0]]['weight'])]
        for i, j in zip(cycle[:-1], cycle[1:]):
            edges.append((i, j))
            scores.append(g[i][j]['weight'])

        i, j = edges[scores.index(min(scores))]
        gc = deepcopy(g)
        gc.remove_edge(i, j)
        gc.add_edge(j, i)
        ngc = len(list(nx.simple_cycles(gc)))
        if ngc < ncycles:
            g.add_edge(j, i, weight=min(scores))
        g.remove_edge(i, j)
        ncycles = ngc
    return g
