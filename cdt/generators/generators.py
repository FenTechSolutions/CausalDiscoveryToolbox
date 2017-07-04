""" Regression and generation functions
Author: Diviyan Kalainathan & Olivier Goudet
Date : 30/06/17
"""

import numpy as np
import tensorflow as tf
import torch as th
from torch.autograd import Variable
from ..utils.Loss import MomentMatchingLoss_th as MomentMatchingLoss, MMD_loss_tf as MMD
from ..utils.Settings import Settings as SETTINGS
from sklearn.linear_model import LassoLars
from sklearn.svm import SVR


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
        out = th.FloatTensor(x.size()[0], int(((x.size()[1]) * (x.size()[1] - 1)) / 2))
        cpt = 0
        for i in range(x.size()[1]):
            for j in range(i+1, x.size()[1]):
                out[:, cpt] = x[:, i] * x[:, j]
                cpt += 1
        # print(int(((x.size()[1])*(x.size()[1]+1))/2)-1, cpt)
        return out

    def forward(self, x=None, n_examples=None, fixed_noise=False):
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
            inputx = th.cat([th.FloatTensor(x), th.FloatTensor(n_examples, 1).fill_(1)], 1)

        inputx = Variable(self.polynomial_features(inputx))
        return self.l(inputx)


class FullGraphPolynomialModel_th(th.nn.Module):
    """ Generate all variables in the graph at once, torch model

    """
    def __init__(self, graph, N):
        """ Initialize the model, build the computation graph

        :param graph: graph to model
        :param N: Number of examples to generate
        """
        super(FullGraphPolynomialModel_th, self).__init__()
        self.graph = graph
        # building the computation graph
        self.graph_variables = []
        self.params = []
        self.N = N
        nodes = self.graph.get_list_nodes()
        while self.graph_variables < len(nodes):
            for var in nodes:
                par = self.graph.get_parents(var)
                if (var not in self.graph_variables and
                    set(par).issubset(self.graph_variables)):
                    # Variable can be generated
                    self.params.append(th.nn.Linear(int((len(par) + 2) * (len(par) + 1) / 2), 1))
                    self.graph_variables.append(par)

    def forward(self, N):
        """ Pass through the generative network

        :return: Generated data
        """
        generated_variables = {}
        for var, layer in zip(self.graph_variables, self.params):
            par = self.graph.get_parents(var)
            if len(par) > 0:
                inputx = th.cat([th.cat([generated_variables[parent] for parent in par], 1),
                                 th.FloatTensor(self.N, 1).normal_(),
                                 th.FloatTensor(self.N, 1).fill_(1)], 1)
            else:
                inputx = th.cat([th.FloatTensor(self.N, 1).normal_(),
                                 th.FloatTensor(self.N, 1).fill_(1)], 1)

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
    """ Initialize a random tensor, normal(0,SETTINGS.init_weights).
        :param size: Size of the tensor
        :return: Tensor
    """
    return tf.random_normal(shape=size, stddev=SETTINGS.init_weights)


class FullGraphPolynomialModel_tf(object):
    def __init__(self, N, graph, list_nodes, run, idx, learning_rate=SETTINGS.learning_rate):
        """ Build the tensorflow graph of the Polynomial generator structure

        :param N: Number of points
        :param graph: Graph to be run
        :param run: number of the run (only for log)
        :param idx: number of the idx (only for log)
        :param learning_rate: learning rate of the optimizer
        """
        super(FullGraphPolynomialModel_tf, self).__init__()
        self.run = run
        self.idx = idx
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
        for var in list_nodes:
            listvariablegraph.append(generated_variables[var])

        self.all_generated_variables = tf.concat(listvariablegraph, 1)
        self.G_dist_loss_xcausesy = MMD(self.all_real_variables, self.all_generated_variables)

        # var_list = theta_G
        self.G_solver_xcausesy = (tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.G_dist_loss_xcausesy,
                                                  var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, verbose=True):
        """ Train the polynomial model by fitting on data using MMD

        :param data: data to fit
        :param verbose: verbose
        :return: None
        """
        for it in range(SETTINGS.nb_epoch_train):
            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.all_real_variables: data}
            )

            if verbose:
                if it % 50 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.idx, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data, verbose=True):
        """ Run the model to generate data and output

        :param data: input data
        :param verbose: verbose
        :return: Generated data
        """

        sumMMD_tr = 0

        for it in range(1):

            MMD_tr, generated_variables = self.sess.run([self.G_dist_loss_xcausesy,
                                                         self.all_generated_variables],
                                                        feed_dict={self.all_real_variables: data})
            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'
                          .format(self.idx, self.run, it, MMD_tr))

        tf.reset_default_graph()

        return generated_variables


def run_graph_polynomial_tf(df_data, graph, idx=0, run=0):
    """ Run the full graph polynomial generator

    :param df_data: data
    :param graph: the graph to model
    :param idx: index (optional, for log purposes)
    :param run: no of run (optional, for log purposes)
    :return: Generated data using the graph structure
    """
    list_nodes = graph.get_list_nodes()
    print(list_nodes)
    df_data = df_data[list_nodes].as_matrix()
    data = df_data.astype('float32')

    if SETTINGS.GPU:
        with tf.device('/gpu:' + str(SETTINGS.gpu_offset + run % SETTINGS.num_gpu)):

            CGNN = FullGraphPolynomialModel_tf(df_data.shape[0], graph, list_nodes, run, idx)
            CGNN.train(data)
            return CGNN.evaluate(data)
    else:
        CGNN = FullGraphPolynomialModel_tf(len(df_data), graph, list_nodes, run, idx)
        CGNN.train(data)
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
    criterion = MomentMatchingLoss(3)
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
    """ Regression and prediction using a lasso

    :param x: data
    :param target: target - effect
    :param causes: causes of the causal mechanism
    :return: regenerated data with the fitted model
    """

    if len(causes) == 0:
        x= np.random.normal(size=(target.shape[0], 1))

    lasso = LassoLars(alpha=1.)  # no regularization
    lasso.fit(x, target)

    return lasso.predict(x)


def support_vector_regressor(x, target, causes):
    """ Regression and prediction using a SVM (rbf)

    :param x: data
    :param target: target - effect
    :param causes: causes of the causal mechanism
    :return: regenerated data with the fitted model
    """
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    if len(causes) == 0:
        x = np.random.normal(size=(target.shape[0], 1))

    return svr_rbf.fit(x, target).predict(x)

