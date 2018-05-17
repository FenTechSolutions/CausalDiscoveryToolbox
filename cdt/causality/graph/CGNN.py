"""Causal Generative Neural Networks.

Author : Olivier Goudet & Diviyan Kalainathan
Ref : Causal Generative Neural Networks (https://arxiv.org/abs/1711.08936)
Date : 09/5/17
"""
import networkx as nx
import numpy as np
import itertools
import warnings
import torch as th
from torch.autograd import Variable
from copy import deepcopy
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
from .model import GraphModel
from ..pairwise.GNN import GNN
from ...utils.loss import MMDloss
from ...utils.Settings import SETTINGS
from ...utils.graph_utils import dagify_min_edge


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class CGNN_block(th.nn.Module):
    """CGNN 'block' which represents a FCM equation between a cause and its parents."""

    def __init__(self, sizes):
        """Init the block with the network sizes."""
        super(CGNN_block, self).__init__()
        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(th.nn.Linear(i, j))
            layers.append(th.nn.ReLU())

        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = th.nn.Sequential(*layers)

    def forward(self, x):
        """Forward through the network."""
        return self.layers(x)


class CGNN_model(th.nn.Module):
    """Class for one CGNN instance."""

    def __init__(self, graph, batch_size, nh=20, gpu=None,
                 gpu_id=0, confounding=False, initial_graph=None, **kwargs):
        """Init the model by creating the blocks and extracting the topological order."""
        super(CGNN_model, self).__init__()
        gpu = SETTINGS.get_default(gpu=gpu)
        device = 'cuda:{}'.format(gpu_id) if gpu else 'cpu'
        nodes = list(graph.nodes())
        self.topological_order = [nodes.index(i) for i in nx.topological_sort(graph)]
        self.adjacency_matrix = nx.adj_matrix(graph, weight=None).todense()
        self.confounding = confounding
        if initial_graph is None:
            initial_graph = graph
            self.i_adj_matrix = self.adjacency_matrix
        else:
            self.i_adj_matrix = nx.adj_matrix(initial_graph, weight=None).todense()
        self.blocks = th.nn.ModuleList()
        self.generated = [None for i in range(self.adjacency_matrix.shape[0])]
        self.noise = th.zeros(batch_size, self.adjacency_matrix.shape[0]).to(device)
        self.corr_noise = dict([[(i, j), th.zeros(batch_size, 1).to(device)] for i, j
                                in zip(*np.nonzero(self.i_adj_matrix)) if i < j])
        self.criterion = MMDloss(batch_size, device=device)
        self.score = th.FloatTensor([0]).to(device)

        for i in range(self.adjacency_matrix.shape[0]):
            if not confounding:
                self.blocks.append(CGNN_block([sum(self.adjacency_matrix[:, i]) + 1, nh, 1]))
            else:
                self.blocks.append(CGNN_block([sum(self.i_adj_matrix[:, i]) +
                                               sum(self.adjacency_matrix[:, i]) + 1, nh, 1]))

    def forward(self):
        """Generate according to the topological order of the graph."""
        self.noise.data.normal_()
        if not self.confounding:
            for i in self.topological_order:
                self.generated[i] = self.blocks[i](th.cat([v for c in [
                                                   [self.generated[j] for j in np.nonzero(self.adjacency_matrix[:, i])[0]],
                                                   [self.noise[:, [i]]]] for v in c], 1))
        else:
            for i in self.topological_order:
                self.generated[i] = self.blocks[i](th.cat([v for c in [
                                                   [self.generated[j] for j in np.nonzero(self.adjacency_matrix[:, i])[0]],
                                                   [self.corr_noise[min(i, j), max(i, j)] for j in np.nonzero(self.i_adj_matrix[:, i])[0]]
                                                   [self.noise[:, [i]]]] for v in c], 1))
        return th.cat(self.generated, 1)

    def run(self, data, train_epochs=1000, test_epochs=1000, verbose=None,
            idx=0, lr=0.01, **kwargs):
        """Run the CGNN on a given graph."""
        verbose = SETTINGS.get_default(verbose=verbose)
        optim = th.optim.Adam(self.parameters(), lr=lr)
        self.score.zero_()

        for epoch in range(train_epochs + test_epochs):
            generated_data = self.forward()
            mmd = self.criterion(generated_data, data)
            if verbose and not epoch % 200:
                print("IDX: {}, MMD Score: {}".format(idx, mmd.item()))
            mmd.backward()
            optim.step()
            if epoch >= test_epochs:
                self.score.add_(mmd.data)

        return self.score.cpu().numpy() / test_epochs


def graph_evaluation(data, graph, gpu=None, gpu_id=0, **kwargs):
    """Evaluate a graph taking account of the hardware."""
    gpu = SETTINGS.get_default(gpu=gpu)
    device = 'cuda:{}'.format(gpu_id) if gpu else 'cpu'
    obs = Variable(th.FloatTensor(data)).to(device)
    cgnn = CGNN_model(graph, data.shape[0], gpu_id=gpu_id, **kwargs).to(device)

    return cgnn.run(obs, **kwargs)


def parallel_graph_evaluation(data, graph, nb_runs=16,
                              nb_jobs=None, **kwargs):
    """Parallelize the various runs of CGNN to evaluate a graph."""
    nb_jobs = SETTINGS.get_default(nb_jobs=nb_jobs)
    if nb_runs == 1:
        return graph_evaluation(data, graph, **kwargs)
    else:
        output = Parallel(n_jobs=nb_jobs)(delayed(graph_evaluation)(data, graph,
                                          idx=run, gpu_id=run % SETTINGS.GPU,
                                          **kwargs) for run in range(nb_runs))
        return np.mean(output)


def hill_climbing(data, graph, **kwargs):
    """Hill Climbing optimization: a greedy exploration algorithm."""
    tested_candidates = [nx.adj_matrix(graph, weight=None)]
    best_score = parallel_graph_evaluation(data, graph, ** kwargs)
    best_candidate = graph
    can_improve = True
    while can_improve:
        can_improve = False
        for (i, j) in best_candidate.edges():
            test_graph = deepcopy(best_candidate)
            test_graph.add_edge(j, i, weight=test_graph[i][j]['weight'])
            test_graph.remove_edge(i, j)
            tadjmat = nx.adj_matrix(test_graph, weight=None)
            if (nx.is_directed_acyclic_graph(test_graph) and not any([(tadjmat != cand).nnz ==
                                                                      0 for cand in tested_candidates])):
                tested_candidates.append(tadjmat)
                score = parallel_graph_evaluation(data, test_graph, **kwargs)
                if score < best_score:
                    can_improve = True
                    best_candidate = test_graph
                    best_score = score
                    break
    return best_candidate


def hill_climbing_with_removal():
    pass


def exploratory_hill_climbing(data, graph, proba=0.1, decay=0.95, max_trials=20, **kwargs):
    """Hill climbing with a bit more exploration."""
    pass


def tabu_search():
    pass


class CGNN(GraphModel):
    """CGNN : Generate the whole causal graph and predict causal directions in the graph.

    Author : Olivier Goudet & Diviyan Kalainathan & Al.
    Ref : Causal Generative Neural Networks (https://arxiv.org/abs/1711.08936)
    """

    def __init__(self):
        """ Initialize the CGNN Model."""
        super(CGNN, self).__init__()

    def create_graph_from_data(self, data, nh=20, nb_runs=16, nb_jobs=None, gpu=None,
                               lr=0.01, train_epochs=1000, test_epochs=1000, verbose=None):
        """Use CGNN to create a graph from scratch."""
        warnings.warn("An exhaustive search of the causal structure of CGNN without"
                      " skeleton is super-exponential in the number of variables.")

        nb_jobs, verbose, gpu = SETTINGS.get_default(('nb_jobs', nb_jobs), ('verbose', verbose), ('gpu', gpu))
        # Building all possible candidates:
        nb_vars = len(list(data.columns))
        data = scale(data.as_matrix()).astype('float32')

        candidates = [np.reshape(np.array(i), (nb_vars, nb_vars)) for i in itertools.product([0, 1], repeat=nb_vars*nb_vars)
                      if (np.trace(np.reshape(np.array(i), (nb_vars, nb_vars))) == 0
                          and nx.is_directed_acyclic_graph(nx.DiGraph(np.reshape(np.array(i), (nb_vars, nb_vars)))))]

        warnings.warn("A total of {} graphs will be evaluated.".format(len(candidates)))
        scores = [parallel_graph_evaluation(data, nx.DiGraph(i), nh=nh, nb_runs=nb_runs, gpu=gpu,
                                            nb_jobs=nb_jobs, lr=lr, train_epochs=train_epochs, test_epochs=test_epochs, verbose=verbose) for i in candidates]
        final_candidate = candidates[scores.index(min(scores))]
        output = np.zeros(final_candidate.shape)

        # Retrieve the confidence score on each edge.
        for (i, j), x in np.ndenumerate(final_candidate):
            if x > 0:
                cand = final_candidate
                cand[i, j] = 0
                output[i, j] = min(scores) - scores[candidates.index(cand)]

        return nx.DiGraph(candidates[output],
                          {idx: i for idx, i in enumerate(data.columns)})

    def orient_directed_graph(self, data, dag, alg='HC', nh=20, nb_runs=16, nb_jobs=None,
                              gpu=None, lr=0.01, train_epochs=1000, test_epochs=1000, verbose=None):
        """Improve a directed acyclic graph using CGNN.

        :param data: data
        :param dag: directed acyclic graph to optimize
        :param alg: type of algorithm
        :param log: Save logs of the execution
        :return: improved directed acyclic graph
        """
        nb_jobs, verbose, gpu = SETTINGS.get_default(('nb_jobs', nb_jobs), ('verbose', verbose), ('gpu', gpu))
        alg_dic = {'HC': hill_climbing, 'HCr': hill_climbing_with_removal,
                   'tabu': tabu_search, 'EHC': exploratory_hill_climbing}
        data = scale(data.as_matrix()).astype('float32')

        return alg_dic[alg](data, dag, nh=nh, nb_runs=nb_runs, gpu=gpu,
                            nb_jobs=nb_jobs, lr=lr, train_epochs=train_epochs,
                            test_epochs=test_epochs, verbose=verbose)

    def orient_undirected_graph(self, data, umg, nh=20, nb_runs=16, nb_jobs=None, gpu=None,
                                lr=0.01, train_epochs=1000, test_epochs=1000, verbose=None, nb_max_runs=16):
        """Orient the undirected graph using GNN and apply CGNN to improve the graph.

        :param data: data
        :param umg: undirected acyclic graph
        :return: directed acyclic graph
        """
        warnings.warn("The pairwise GNN model is computed on each edge of the UMG "
                      "to initialize the model and start CGNN with a DAG")
        if verbose:
            raise ValueError
        nb_jobs, verbose, gpu = SETTINGS.get_default(('nb_jobs', nb_jobs), ('verbose', verbose), ('gpu', gpu))
        gnn = GNN(nh=nh, lr=lr)

        og = gnn.orient_graph(data, umg, nb_runs=nb_runs, nb_max_runs=nb_max_runs,
                              nb_jobs=nb_jobs, train_epochs=train_epochs,
                              test_epochs=test_epochs, verbose=verbose, gpu=gpu)  # Pairwise method
        # print(nx.adj_matrix(og).todense().shape)

        dag = dagify_min_edge(og)
        # print(nx.adj_matrix(dag).todense().shape)

        return self.orient_directed_graph(data, dag,  nh=nh, nb_runs=nb_runs, gpu=gpu,
                                          nb_jobs=nb_jobs, lr=lr, train_epochs=train_epochs,
                                          test_epochs=test_epochs, verbose=verbose)
