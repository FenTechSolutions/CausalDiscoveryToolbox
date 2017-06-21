"""
Build random graphs based on unlabelled data
Author : Diviyan Kalainathan & Olivier Goudet
Date: 17/6/17
"""
import numpy as np
import pandas as pd
import torch as th
from torch.autograd import Variable
from ..utils.Graph import DirectedGraph
from ..utils.loss import MomentMatchingLoss


class PolynomialModel(th.nn.Module):
    """ Model for multi-polynomial regression
    """

    def __init__(self, rank, degree=2):
        """ Initialize the model

        :param rank: number of causes to be fitted
        :param degree: degree of the polynomial function
        Warning : only degree == 2 has been implemented
        """
        assert degree == 2
        super(PolynomialModel, self).__init__()
        self.l = th.nn.Linear(degree ** (rank + 2), 1, bias=False)

    def polynomial_features(self, x):
        """ Featurize data using a matrix multiplication trick

        :param x: unfeaturized data
        :return: featurized data for polynomial regression of degree=2
        """
        return th.cat([(x[i].t() @ x[i]).view(1, -1) for i in range(x.size()[0])], 0)

    def forward(self, x=None, n_examples=None):
        """ Featurize and compute output

        :param x: input data
        :param n_examples: number of examples (for the case of no input data)
        :return: predicted data using weights
        """

        if x is not None:
            x = th.from_numpy(x)
            inputx = th.cat([x,
                             th.FloatTensor(n_examples, 1).normal_(),
                             th.FloatTensor(n_examples, 1).fill_(1)], 1)
        else:
            inputx = th.cat([th.FloatTensor(n_examples, 1).normal_(),
                             th.FloatTensor(n_examples, 1).fill_(1)], 1)
        inputx = Variable(self.polynomial_features(inputx))
        return self.l(inputx)


def polynomial_regressor(df, causes, target, train_epochs=30, verbose=True):
    """ Regress data using a polynomial regressor of degree 2

    :param df: data
    :param causes: list of parent nodes
    :param target: target node
    :param train_epochs: number of train epochs
    :param verbose: verbose
    :return: generated data
    """
    if len(causes) > 0:
        x = df[causes].as_matrix()
    else:
        causes = None
    n_ex = len(df)
    target = df[target].as_matrix()
    model = PolynomialModel(len(causes), degree=2)
    criterion = MomentMatchingLoss(4)
    optimizer = th.optim.SGD(model.parameters())

    for epoch in range(train_epochs):
        y_tr = model(x)
        loss = criterion(y_tr, target)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print('Epoch : {} ; Loss: {}'.format(epoch, loss.numpy()))

    return model(x)


class RandomGraphFromData(object):
    """ Generate a random graph out of data

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
        self.simulator = simulator
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
            corr = np.fill_diagonal(corr, 0.)
        else:
            corr = np.zeros((len(self.data.columns), len(self.data.columns)))
            for idxi, i in enumerate(self.data.columns[:-1]):
                for idxj, j in enumerate(self.data.columns[idxi + 1:]):
                    corr[idxi, idxj] = np.absolute(self.criterion(self.data[i], self.data[j]))
                    corr[idxj, idxi] = corr[idxi, idxj]

        print(corr)
        self.llinks = [(self.data.columns[i], self.data.columns[j])
                       for i in range(len(self.data.columns) - 1)
                       for j in range(i + 1, len(self.data.columns)) if corr[i, j] > threshold]


    def generate_graph(self, draw_proba=.8):
        """ Generate random graph out of the data

        :param draw_proba:
        :return:
        """
        # Find dependencies
        if self.llinks is None:
            self.find_dependencies()

        # Draw random number of edges out of the dependent edges and create an acyclic graph
        graph = DirectedGraph()

        for link in self.llinks:
            if np.random.uniform() < draw_proba:
                if np.random.uniform() < 0.5:
                    link = list(reversed(link))
                else:
                    link = list(link)

                graph.add(link[0], link[1], 1)

        graph.remove_cycles()

        # Resimulation of variables
        generated_variables = []
        nodes = graph.get_list_nodes()
        self.resimulated_data = pd.DataFrame()

        # Regress using a y=P(Xc,E)= Sum_i,j^d(_alphaij*(X_1+..+X_c)^i*E^j) model & re-simulate data
        while len(generated_variables) < len(nodes):
            for var in nodes:
                par = graph.get_parents(var)
                if (var not in generated_variables and
                        set(par).issubset(generated_variables)):
                    # Variable can be generated
                    self.resimulated_data[var] = self.simulator(self.data, par, var)

        return graph, self.resimulated_data
