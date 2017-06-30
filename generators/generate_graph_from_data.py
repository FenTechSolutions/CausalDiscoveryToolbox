"""
Build random graphs based on unlabelled data
Author : Diviyan Kalainathan & Olivier Goudet
Date: 17/6/17
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import torch as th
from torch.autograd import Variable
from copy import deepcopy
from ..utils.Graph import DirectedGraph
from ..utils.loss import MomentMatchingLoss_th as MomentMatchingLoss, MMD_loss as MMD
from ..utils.SETTINGS import CGNN_SETTINGS as SETTINGS
from sklearn.linear_model import LassoLars
from sklearn.svm import SVR
import matplotlib.pyplot as plt


class PolynomialModel(th.nn.Module):
    """ Model for multi-polynomial regression - Generation one-by-one
    """

    def __init__(self, rank, degree=2):
        """ Initialize the model

        :param rank: number of causes to be fitted
        :param degree: degree of the polynomial function
        Warning : only degree == 2 has been implemented
        """
        assert degree == 2
        super(PolynomialModel, self).__init__()
        self.l = th.nn.Linear(
            int(((rank + 2) * (rank + 1)) / 2), 1, bias=False)

    def polynomial_features(self, x):
        """ Featurize data using a matrix multiplication trick

        :param x: unfeaturized data
        :return: featurized data for polynomial regression of degree=2
        """
        # return th.cat([(x[i].view(-1, 1) @ x[i].view(1, -1)).view(1, -1) for
        # i in range(x.size()[0])], 0) WRONG
        out = th.FloatTensor(x.size()[0], int(
            ((x.size()[1]) * (x.size()[1] - 1)) / 2))
        cpt = 0
        for i in range(x.size()[1]):
            for j in range(i+1, x.size()[1]):
                out[:, cpt] = x[:, i] * x[:, j]
                cpt += 1
        # print(int(((x.size()[1])*(x.size()[1]+1))/2)-1, cpt)
        return out

    def forward(self, x=None, n_examples=None, fixed_noise=True):
        """ Featurize and compute output

        :param x: input data
        :param n_examples: number of examples (for the case of no input data)
        :return: predicted data using weights
        """
        if not fixed_noise:
            inputx = th.cat([th.FloatTensor(n_examples, 1).fill_(1),
                             th.FloatTensor(n_examples, 1).normal_()], 1)
            if x is not None:
                x = th.FloatTensor(x)
                inputx = th.cat([x, inputx], 1)
        else:
            inputx = th.cat([x, th.FloatTensor(n_examples, 1).fill_(1)], 1)

        inputx = Variable(self.polynomial_features(inputx))
        return self.l(inputx)


class FullGraphPolynomialModel(th.nn.Module):
    def __init__(self, graph):
        super(FullGraphPolynomialModel, self).__init__()
        self.graph = graph
        # building the computation graph
        self.graph_variables = []
        self.params = []
        nodes = self.graph.get_list_nodes()
        while self.graph_variables < len(nodes):
            for var in nodes:
                par = self.graph.get_parents(var)
                if (var not in self.graph_variables and
                    set(par).issubset(self.graph_variables)):
                    # Variable can be generated
                    self.params.append(th.nn.Linear(int((len(par) + 2) * (len(par) + 1) / 2), 1))
                    self.graph_variables.append(par)

    def forward(self, data):
        generated_variables = {}
        for var, layer in zip(self.graph_variables, self.params):
            par = self.graph.get_parents(var)
            if len(par) > 0:
                inputx = th.cat([th.from_numpy(np.array(data[par].as_matrix(), dtype=np.float32)),
                                 th.FloatTensor(len(data), 1).normal_(),
                                 th.FloatTensor(len(data), 1).fill_(1)], 1)
            else:
                inputx = th.cat([th.FloatTensor(len(data), 1).normal_(),
                                 th.FloatTensor(len(data), 1).fill_(1)], 1)

            x = []
            for i in range(len(par) + 2):
                for j in range(i + 1, len(par) + 2):
                    x.append(inputx[i] * inputx[j])

            inputx = Variable(th.cat(x, 1))
            generated_variables[var] = layer(inputx)

        output = []
        for v in self.graph.get_list_nodes():
            output.append(generated_variables[v])

        return th.cat(output, 1)


def init(size):
    return tf.random_normal(shape=size, stddev=SETTINGS.init_weights)

class CGNN_graph(object):
    def __init__(self, N, graph, list_nodes, run, pair, learning_rate=SETTINGS.learning_rate):
        """
        Build the tensorflow graph,
        For a given structure
        """
        self.run = run
        self.pair = pair
        n_var = len(list_nodes)

        self.all_real_variables = tf.placeholder(tf.float32, shape=[None, n_var])
        alpha = tf.Variable(init([1, 1]))
        generated_variables = {}
        theta_G = [alpha]

        while len(generated_variables) < n_var:
            for var in list_nodes:
                # Check if all parents are generated
                par = graph.get_parents(var)

                if (var not in generated_variables and
                        set(par).issubset(generated_variables)):

                    # Generate the variable
                    W_in = tf.Variable(init([int((len(par) + 2) * (len(par) + 1) / 2), 1]))

                    input_v = []
                    input_v.append(tf.ones([N, 1]))
                    for i in par:
                        input_v.append(generated_variables[i])
                    input_v.append(tf.random_normal([N, 1], mean=0, stddev=1))

                    out_v = 0
                    cpt = 0
                    for i in range(len(par) + 2):
                        for j in range(i + 1, len(par) + 2):
                            out_v += W_in[cpt] * tf.multiply(input_v[i], input_v[j])
                            cpt += 1

                    generated_variables[var] = out_v
                    theta_G.extend([W_in])


        listvariablegraph = []
        for var in list_nodes[:2]:
            listvariablegraph.append(generated_variables[var])

        all_generated_variables = tf.concat(listvariablegraph, 1)

        self.G_dist_loss_xcausesy = MMD(self.all_real_variables[:,:2], all_generated_variables)

        # var_list = theta_G
        self.G_solver_xcausesy = (tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.G_dist_loss_xcausesy,
                                                  var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, verbose=True):
        for it in range(SETTINGS.nb_epoch_train):
            print(it)
            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.all_real_variables: data}
            )

            if verbose:
                if (it % 1 == 0):
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.pair, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data, verbose=True):

        sumMMD_tr = 0

        for it in range(SETTINGS.nb_epoch_test):

            MMD_tr = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={
                self.all_real_variables: data})

            sumMMD_tr += MMD_tr[0]

            if verbose:
                if (it % 100 == 0):
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'
                          .format(self.pair, self.run, it, MMD_tr[0]))

        tf.reset_default_graph()

        return sumMMD_tr / SETTINGS.nb_epoch_test


def run_graph_polynomial(df_data, graph, idx, run):
    list_nodes = graph.get_list_nodes()
    df_data = df_data[list_nodes].as_matrix()
    data = df_data.astype('float32')
    print('OK')

    if SETTINGS.GPU:
        with tf.device('/gpu:' + str(SETTINGS.gpu_offset + run % SETTINGS.num_gpu)):

            CGNN = CGNN_graph(df_data.shape[0], graph, list_nodes, run, idx)
            print('OK')
            CGNN.train(data)
            print('OKtrain')
            return CGNN.evaluate(data)
    else:
        CGNN = CGNN_graph(len(df_data), graph, list_nodes, run, idx)
        print('OK')
        CGNN.train(data)
        print('OK')
        return CGNN.evaluate(data)


def polynomial_regressor(x, target, causes, train_epochs=1000, fixed_noise=False, verbose=True):
    """ Regress data using a polynomial regressor of degree 2

    :param x: parents data
    :param target: target data
    :param causes: list of parent nodes
    :param train_epochs: number of train epochs
    :param fixed_noise : If the noise in the generation is fixed or not.
    :param verbose: verbose
    :return: generated data
    """

    n_ex = target.shape[0]
    if len(causes) == 0:
        causes = []
        x = None
        if fixed_noise:
            x = th.FloatTensor(n_ex, 1).normal_()
    elif fixed_noise:
        x = th.FloatTensor(x)
        x = th.cat([x, th.FloatTensor(n_ex, 1).normal_()], 1)
    target = Variable(th.FloatTensor(target))
    model = PolynomialModel(len(causes), degree=2)
    if SETTINGS.GPU:
        model.cuda()
        target.cuda()
        if x is not None:
            x.cuda()
    criterion = MMD(n_ex) #MomentMatchingLoss(3)
    optimizer = th.optim.Adam(model.parameters(), lr=10e-3)

    for epoch in range(train_epochs):
        y_tr = model(x, n_ex, fixed_noise=fixed_noise)
        loss = criterion(y_tr, target)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 50 == 0:
            print('Epoch : {} ; Loss: {}'.format(epoch, loss.data.numpy()))

    return model(x, n_ex).data.numpy()


def linear_regressor(x, target, causes):

    if len(causes) == 0:
        x= np.random.normal(size=(target.shape[0], 1))

    lasso = LassoLars(alpha=1.)  # no regularization
    lasso.fit(x, target)

    return lasso.predict(x)


def support_vector_regressor(x, target, causes):
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    if len(causes) == 0:
        x = np.random.normal(size=(target.shape[0], 1))

    return svr_rbf.fit(x, target).predict(x)


class RandomGraphFromData(object):
    """ Generate a random graph out of data : produce a random graph and make statistics fit to the data

    """

    def __init__(self, df_data, simulator=polynomial_regressor,  datatype='Numerical'):
        """

        :param df_data:
        :param simulator:
        :param datatype:
        """
        super(RandomGraphFromData, self).__init__()
        self.data = df_data
        self.resimulated_data = None
        self.matrix_criterion = None
        self.llinks = None
        # self.simulator = simulator
        try:
            assert datatype == 'Numerical'
            self.criterion = np.corrcoef
            self.matrix_criterion = True
        except AssertionError:
            print('Not Yet Implemented')
            raise NotImplementedError

    def find_dependencies(self, threshold=0.05):
        """ Find dependencies in the dataset out of the dataset

        :param threshold:
        """
        if self.matrix_criterion:
            corr = np.absolute(self.criterion(self.data.as_matrix()))
            np.fill_diagonal(corr, 0.)
        else:
            corr = np.zeros((len(self.data.columns), len(self.data.columns)))
            for idxi, i in enumerate(self.data.columns[:-1]):
                for idxj, j in enumerate(self.data.columns[idxi + 1:]):
                    corr[idxi, idxj] = np.absolute(
                        self.criterion(self.data[i], self.data[j]))
                    corr[idxj, idxi] = corr[idxi, idxj]

        self.llinks = [(self.data.columns[i], self.data.columns[j])
                       for i in range(len(self.data.columns) - 1)
                       for j in range(i + 1, len(self.data.columns)) if corr[i, j] > threshold]

    def generate_graph(self, draw_proba=.2):
        """ Generate random graph out of the data

        :param draw_proba:
        :return:
        """
        # Find dependencies
        if self.llinks is None:
            self.find_dependencies()

        # Draw random number of edges out of the dependent edges and create an
        # acyclic graph
        graph = DirectedGraph()

        for link in self.llinks:
            if np.random.uniform() < draw_proba:
                if np.random.uniform() < 0.5:
                    link = list(reversed(link))
                else:
                    link = list(link)

                # Test if adding the link does not create a cycle
                if not deepcopy(graph).add(link[0], link[1], 1).is_cyclic():
                    graph.add(link[0], link[1], 1)
                elif not deepcopy(graph).add(link[1], link[0], 1).is_cyclic():
                    # Test if we can add the link in the other direction
                    graph.add(link[1], link[0], 1)

        graph.remove_cycles()
        print(graph.is_cyclic(), graph.cycles())
        print('Adjacency matrix : {}'.format(graph.get_adjacency_matrix()))
        print('Number of edges : {}'.format(
            len(graph.get_list_edges(return_weights=False))))
        print("Beginning random graph build")
        print("Graph generated, passing to data generation!")
        # Resimulation of variables
        # generated_variables = {}
        nodes = graph.get_list_nodes()

        # Regress using a y=P(Xc,E)= Sum_i,j^d(_alphaij*(X_1+..+X_c)^i*E^j) model & re-simulate data
        # run_graph_polynomial(self.data, graph,0,0)
        print('OK')
        generated_variables = run_graph_polynomial(self.data, graph, 0, 0)

        # while len(generated_variables) < len(nodes):
        #     for var in nodes:
        #         par = graph.get_parents(var)
        #         if (var not in generated_variables and
        #                 set(par).issubset(generated_variables)):
        #             # Variable can be generated
        #             if len(par) == 0:
        #                 generated_variables[var] = self.data[var]
        #             else:
        #                 generated_variables[var] = self.simulator(pd.DataFrame(generated_variables)[
        #                                          par].as_matrix(), self.data[var].as_matrix(), par).reshape(-1)
        #
        #                 if len(par)>0:
        #                     plt.scatter(self.data[par[0]], self.data[var], alpha=0.2)
        #                     plt.scatter(generated_variables[par[0]], generated_variables[var], alpha=0.2)
        #                     plt.show()

        return graph, pd.DataFrame(generated_variables, columns=list_nodes)
