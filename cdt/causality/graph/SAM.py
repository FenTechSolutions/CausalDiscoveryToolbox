"""Structural Agnostic Model.

Author: Diviyan Kalainathan, Olivier Goudet
Date: 09/3/2018
"""

import math
import torch as th
import networkx as nx
from torch.autograd import Variable
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from .model import GraphModel
from ...utils.Settings import SETTINGS


class CNormalized_Linear(th.nn.Module):
    """Linear layer with column-wise normalized input matrix."""

    def __init__(self, in_features, out_features, bias=False):
        """Initialize the layer."""
        super(CNormalized_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = th.nn.Parameter(th.Tensor(out_features, in_features))
        if bias:
            self.bias = th.nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """Feed-forward through the network."""
        return th.nn.functional.linear(input, self.weight.div(self.weight.pow(2).sum(0).sqrt()))

    def __repr__(self):
        """For print purposes."""
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class SAM_discriminator(th.nn.Module):
    """Discriminator for the SAM model."""

    def __init__(self, sizes, zero_components=[], **kwargs):
        """Init the SAM discriminator."""
        super(SAM_discriminator, self).__init__()
        self.sht = kwargs.get('shortcut', False)
        activation_function = kwargs.get('activation_function', th.nn.ReLU)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        dropout = kwargs.get("dropout", 0.)

        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(th.nn.Linear(i, j))
            if batch_norm:
                layers.append(th.nn.BatchNorm1d(j))
            if dropout != 0.:
                layers.append(th.nn.Dropout(p=dropout))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))

        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = th.nn.Sequential(*layers)
        # print(self.layers)

    def forward(self, x):
        """Feed-forward the model."""
        return self.layers(x)


class SAM_block(th.nn.Module):
    """SAM-Block: conditional generator.

    Generates one variable while selecting the parents. Uses filters to do so.
    One fixed filter and one with parameters on order to keep a fixed skeleton.
    """

    def __init__(self, sizes, zero_components=[], **kwargs):
        """Initialize a generator."""
        super(SAM_block, self).__init__()
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        activation_function = kwargs.get('activation_function', th.nn.Tanh)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(CNormalized_Linear(i, j))
            if batch_norm:
                layers.append(th.nn.BatchNorm1d(j))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))

        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = th.nn.Sequential(*layers)

        # Filtering the unconnected nodes.
        self._filter = th.ones(1, sizes[0])
        for i in zero_components:
            self._filter[:, i].zero_()

        self._filter = Variable(self._filter, requires_grad=False)
        self.fs_filter = th.nn.Parameter(self._filter.data)

        if gpu:
            self._filter = self._filter.cuda(gpu_no)

    def forward(self, x):
        """Feed-forward the model."""
        return self.layers(x * (self._filter *
                                self.fs_filter).expand_as(x))


class SAM_generators(th.nn.Module):
    """Ensemble of all the generators."""

    def __init__(self, data_shape, zero_components, nh=200, batch_size=-1, **kwargs):
        """Init the model."""
        super(SAM_generators, self).__init__()
        if batch_size == -1:
            batch_size = data_shape[0]
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        rows, self.cols = data_shape

        # building the computation graph
        self.noise = [Variable(th.FloatTensor(batch_size, 1))
                      for i in range(self.cols)]
        if gpu:
            self.noise = [i.cuda(gpu_no) for i in self.noise]
        self.blocks = th.nn.ModuleList()

        # Init all the blocks
        for i in range(self.cols):
            self.blocks.append(SAM_block(
                [self.cols + 1, nh, 1], zero_components[i], **kwargs))

    def forward(self, x):
        """Feed-forward the model."""
        for i in self.noise:
            i.data.normal_()

        self.generated_variables = [self.blocks[i](
            th.cat([x, self.noise[i]], 1)) for i in range(self.cols)]
        return self.generated_variables


def plot_curves(i_batch, adv_loss, gen_loss, l1_reg, cols):
    """Plot SAM's various losses."""
    from matplotlib import pyplot as plt
    if i_batch == 0:
        try:
            ax.clear()
            ax.plot(range(len(adv_plt)), adv_plt, "r-",
                    linewidth=1.5, markersize=4,
                    label="Discriminator")
            ax.plot(range(len(adv_plt)), gen_plt, "g-", linewidth=1.5,
                    markersize=4, label="Generators")
            ax.plot(range(len(adv_plt)), l1_plt, "b-",
                    linewidth=1.5, markersize=4,
                    label="L1-Regularization")
            plt.legend()

            adv_plt.append(adv_loss.cpu().data[0])
            gen_plt.append(gen_loss.cpu().data[0] / cols)
            l1_plt.append(l1_reg.cpu().data[0])

            plt.pause(0.0001)

        except NameError:
            plt.ion()
            fig, ax = plt.figure()
            plt.xlabel("Epoch")
            plt.ylabel("Losses")

            plt.pause(0.0001)

            adv_plt = [adv_loss.cpu().data[0]]
            gen_plt = [gen_loss.cpu().data[0] / cols]
            l1_plt = [l1_reg.cpu().data[0]]

    else:
        adv_plt.append(adv_loss.cpu().data[0])
        gen_plt.append(gen_loss.cpu().data[0] / cols)
        l1_plt.append(l1_reg.cpu().data[0])


def plot_gen(epoch, batch, generated_variables, pairs_to_plot=[[0, 1]]):
    """Plot generated pairs of variables."""
    from matplotlib import pyplot as plt
    if epoch == 0:
        plt.ion()
    plt.clf()
    for (i, j) in pairs_to_plot:

        plt.scatter(generated_variables[i].data.cpu().numpy(
        ), batch.data.cpu().numpy()[:, j], label="Y -> X")
        plt.scatter(batch.data.cpu().numpy()[
            :, i], generated_variables[j].data.cpu().numpy(), label="X -> Y")

        plt.scatter(batch.data.cpu().numpy()[:, i], batch.data.cpu().numpy()[
            :, j], label="original data")
        plt.legend()

    plt.pause(0.01)


def run_SAM(df_data, skeleton=None, **kwargs):
    """Execute the SAM model.

    :param df_data: Input data; either np.array or pd.DataFrame
    """
    gpu = kwargs.get('gpu', False)
    gpu_no = kwargs.get('gpu_no', 0)

    train_epochs = kwargs.get('train_epochs', 1000)
    test_epochs = kwargs.get('test_epochs', 1000)
    batch_size = kwargs.get('batch_size', -1)

    lr_gen = kwargs.get('lr_gen', 0.1)
    lr_disc = kwargs.get('lr_disc', lr_gen)
    verbose = kwargs.get('verbose', True)
    regul_param = kwargs.get('regul_param', 0.1)
    dnh = kwargs.get('dnh', None)
    plot = kwargs.get("plot", False)
    plot_generated_pair = kwargs.get("plot_generated_pair", False)

    d_str = "Epoch: {} -- Disc: {} -- Gen: {} -- L1: {}"
    try:
        list_nodes = list(df_data.columns)
        df_data = (df_data[list_nodes]).as_matrix()
    except AttributeError:
        list_nodes = list(range(df_data.shape[1]))
    data = df_data.astype('float32')
    data = th.from_numpy(data)
    if batch_size == -1:
        batch_size = data.shape[0]
    rows, cols = data.size()

    # Get the list of indexes to ignore
    if skeleton is not None:
        zero_components = [[] for i in range(cols)]
        skel = nx.adj_matrix(skeleton, weight=None)
        for i, j in zip(*((1-skel).nonzero())):
            zero_components[j].append(i)
    else:
        zero_components = [[i] for i in range(cols)]
    sam = SAM_generators((rows, cols), zero_components, batch_norm=True, **kwargs)

    # Begin UGLY
    activation_function = kwargs.get('activation_function', th.nn.Tanh)
    try:
        del kwargs["activation_function"]
    except KeyError:
        pass
    discriminator_sam = SAM_discriminator(
        [cols, dnh, dnh, 1], batch_norm=True,
        activation_function=th.nn.LeakyReLU,
        activation_argument=0.2, **kwargs)
    kwargs["activation_function"] = activation_function
    # End of UGLY

    if gpu:
        sam = sam.cuda(gpu_no)
        discriminator_sam = discriminator_sam.cuda(gpu_no)
        data = data.cuda(gpu_no)

    # Select parameters to optimize : ignore the non connected nodes
    criterion = th.nn.BCEWithLogitsLoss()
    g_optimizer = th.optim.Adam(sam.parameters(), lr=lr_gen)
    d_optimizer = th.optim.Adam(
        discriminator_sam.parameters(), lr=lr_disc)

    true_variable = Variable(
        th.ones(batch_size, 1), requires_grad=False)
    false_variable = Variable(
        th.zeros(batch_size, 1), requires_grad=False)
    causal_filters = th.zeros(data.shape[1], data.shape[1])

    if gpu:
        true_variable = true_variable.cuda(gpu_no)
        false_variable = false_variable.cuda(gpu_no)
        causal_filters = causal_filters.cuda(gpu_no)

    data_iterator = DataLoader(data, batch_size=batch_size, shuffle=True)

    # TRAIN
    for epoch in range(train_epochs + test_epochs):
        for i_batch, batch in enumerate(data_iterator):
            batch = Variable(batch)
            batch_vectors = [batch[:, [i]] for i in range(cols)]

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # Train the discriminator
            generated_variables = sam(batch)
            disc_losses = []
            gen_losses = []

            for i in range(cols):
                generator_output = th.cat([v for c in [batch_vectors[: i], [
                    generated_variables[i]],
                    batch_vectors[i + 1:]] for v in c], 1)
                # 1. Train discriminator on fake
                disc_output_detached = discriminator_sam(
                    generator_output.detach())
                disc_output = discriminator_sam(generator_output)
                disc_losses.append(
                    criterion(disc_output_detached, false_variable))

                # 2. Train the generator :
                gen_losses.append(criterion(disc_output, true_variable))

            true_output = discriminator_sam(batch)
            adv_loss = sum(disc_losses)/cols + \
                criterion(true_output, true_variable)
            gen_loss = sum(gen_losses)

            adv_loss.backward()
            d_optimizer.step()

            # 3. Compute filter regularization
            filters = th.stack(
                [i.fs_filter[0, :-1].abs() for i in sam.blocks], 1)
            l1_reg = regul_param * filters.sum()
            loss = gen_loss + l1_reg

            if verbose and epoch % 200 == 0 and i_batch == 0:

                print(str(i) + " " + d_str.format(epoch,
                                                  adv_loss.item(),
                                                  gen_loss.item() / cols,
                                                  l1_reg.item()))
            loss.backward()

            # STORE ASSYMETRY values for output
            if epoch > train_epochs:
                causal_filters.add_(filters.data)
            g_optimizer.step()

            if plot:
                plot_curves(i_batch, adv_loss, gen_loss, l1_reg, cols)

            if plot_generated_pair and epoch % 200 == 0:
                plot_gen(epoch, batch, generated_variables)

    return causal_filters.div_(test_epochs).cpu().numpy()


class SAM(GraphModel):
    """Structural Agnostic Model."""

    def __init__(self, lr=0.1, dlr=0.1, l1=0.1, nh=200, dnh=200,
                 train_epochs=1000, test_epochs=1000, batchsize=-1):
        """Init and parametrize the SAM model.

        :param lr: Learning rate of the generators
        :param dlr: Learning rate of the discriminator
        :param l1: L1 penalization on the causal filters
        :param nh: Number of hidden units in the generators' hidden layers
        :param dnh: Number of hidden units in the discriminator's hidden layer$
        :param train_epochs: Number of training epochs
        :param test_epochs: Number of test epochs (saving and averaging the causal filters)
        :param batchsize: Size of the batches to be fed to the SAM model.
        """
        super(SAM, self).__init__()
        self.lr = lr
        self.dlr = dlr
        self.l1 = l1
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.batchsize = batchsize

    def predict(self, data, skeleton=None, nruns=6, njobs=None, gpus=0, verbose=None,
                plot=False, plot_generated_pair=False, return_list_results=False):
        """Execute SAM on a dataset given a skeleton or not.

        :param data: Observational data for estimation of causal relationships by SAM
        :param skeleton: A priori knowledge about the causal relationships as an adjacency matrix.
                         Can be fed either directed or undirected links.
        :param nruns: Number of runs to be made for causal estimation.
                      Recommended: >=12 for optimal performance.
        :param njobs: Numbers of jobs to be run in Parallel.
                      Recommended: 1 if no GPU available, 2*number of GPUs else.
        :param gpus: Number of available GPUs for the algorithm.
        :param verbose: verbose mode
        :param plot: Plot losses interactively. Not recommended if nruns>1
        :param plot_generated_pair: plots a generated pair interactively.  Not recommended if nruns>1
        :return: Adjacency matrix (A) of the graph estimated by SAM,
                A[i,j] is the term of the ith variable for the jth generator.
        """
        verbose, njobs = SETTINGS.get_default(('verbose', verbose), ('nb_jobs', njobs))
        if njobs != 1:
            list_out = Parallel(n_jobs=njobs)(delayed(run_SAM)(data,
                                                               skeleton=skeleton,
                                                               lr_gen=self.lr, lr_disc=self.dlr,
                                                               regul_param=self.l1, nh=self.nh, dnh=self.dnh,
                                                               gpu=bool(gpus), train_epochs=self.train,
                                                               test_epochs=self.test, batch_size=self.batchsize,
                                                               plot=plot, verbose=verbose, gpu_no=idx % max(gpus, 1))
                                              for idx in range(nruns))
        else:
            list_out = [run_SAM(data, skeleton=skeleton,
                                lr_gen=self.lr, lr_disc=self.dlr,
                                regul_param=self.l1, nh=self.nh, dnh=self.dnh,
                                gpu=bool(gpus), train_epochs=self.train,
                                test_epochs=self.test, batch_size=self.batchsize,
                                plot=plot, verbose=verbose, gpu_no=0)
                        for idx in range(nruns)]
        if return_list_results:
            return list_out
        else:
            W = list_out[0]
            for w in list_out[1:]:
                W += w
            W /= nruns
        return nx.relabel_nodes(nx.DiGraph(W), {idx: i for idx, i in enumerate(data.columns)})

    def orient_directed_graph(self, *args, **kwargs):
        """Orient a (partially directed) graph."""
        return self.predict(*args, **kwargs)

    def orient_undirected_graph(self, *args, **kwargs):
        """Orient a undirected graph."""
        return self.predict(*args, **kwargs)

    def create_graph_from_data(self, *args, **kwargs):
        """Estimate a causal graph out of observational data."""
        return self.predict(*args, **kwargs)
